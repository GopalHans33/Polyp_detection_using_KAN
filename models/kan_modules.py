"""
models/kan_modules.py  —  KA-ResUNet++
========================================
Core KAN (Kolmogorov-Arnold Network) components.
Self-contained: No external dependencies (timm) required.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers (Replacements for timm to ensure standalone capability)
# ══════════════════════════════════════════════════════════════════════════════

def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Standard truncation helper
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) for residual connections."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# ══════════════════════════════════════════════════════════════════════════════
#  KANLinear — The Core Logic
# ══════════════════════════════════════════════════════════════════════════════

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        # Initialize Grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise  = scale_noise
        self.scale_base   = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps      = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output   = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


# ══════════════════════════════════════════════════════════════════════════════
#  FastKANConvLayer + Basis Functions
# ══════════════════════════════════════════════════════════════════════════════

class SplineConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, init_scale=0.1, **kw):
        self.init_scale = init_scale
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, **kw)

    def reset_parameters(self):
        trunc_normal_(self.weight, mean=0, std=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class BSplineFunction(nn.Module):
    def __init__(self, grid_min=-2., grid_max=2., degree=3, num_basis=8):
        super().__init__()
        self.degree    = degree
        self.num_basis = num_basis
        self.knots     = torch.linspace(grid_min, grid_max, num_basis + degree + 1)

    def basis_function(self, i, k, t):
        if k == 0:
            return ((self.knots[i] <= t) & (t < self.knots[i + 1])).float()
        left_num  = (t - self.knots[i]) * self.basis_function(i, k - 1, t)
        left_den  = self.knots[i + k] - self.knots[i]
        left      = left_num / left_den if left_den != 0 else 0
        right_num = (self.knots[i + k + 1] - t) * self.basis_function(i + 1, k - 1, t)
        right_den = self.knots[i + k + 1] - self.knots[i + 1]
        right     = right_num / right_den if right_den != 0 else 0
        return left + right

    def forward(self, x):
        x = x.squeeze()
        # For simplicity, we implement the forward loop. 
        # Note: Ideally precompute knots on device.
        if self.knots.device != x.device:
            self.knots = self.knots.to(x.device)
            
        basis_functions = torch.stack(
            [self.basis_function(i, self.degree, x) for i in range(self.num_basis)], dim=-1
        )
        return basis_functions

class FastKANConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, grid_min=-2., grid_max=2.,
                 num_grids=4, use_base_update=True, base_activation=F.silu,
                 spline_weight_init_scale=0.1, kan_type="BSpline"):
        super().__init__()
        
        self.rbf = BSplineFunction(grid_min, grid_max, 4, num_grids) # Default to BSpline

        self.spline_conv = SplineConv2D(
            in_channels * num_grids, out_channels, kernel_size,
            stride, padding, dilation, groups, bias,
            init_scale=spline_weight_init_scale
        )
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x_rbf = self.rbf(x.view(B, C, -1)).view(B, C, H, W, -1)
        x_rbf = x_rbf.permute(0, 4, 1, 2, 3).contiguous().view(B, -1, H, W)
        ret = self.spline_conv(x_rbf)
        if self.use_base_update:
            base = self.base_conv(self.base_activation(x))
            ret  = ret + base
        return ret


# ══════════════════════════════════════════════════════════════════════════════
#  Building Blocks (DWConv, KANLayer, KANBlock)
# ══════════════════════════════════════════════════════════════════════════════

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn     = nn.BatchNorm2d(dim)
        self.relu   = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        kan_kwargs = dict(
            grid_size=5, spline_order=3, scale_noise=0.1,
            scale_base=1.0, scale_spline=1.0,
            base_activation=nn.SiLU,
            grid_eps=0.02, grid_range=[-1, 1],
        )

        if not no_kan:
            self.fc1 = KANLinear(in_features,      hidden_features, **kan_kwargs)
            self.fc2 = KANLinear(hidden_features,   out_features,   **kan_kwargs)
            self.fc3 = KANLinear(hidden_features,   out_features,   **kan_kwargs)
        else:
            self.fc1 = nn.Linear(in_features,      hidden_features)
            self.fc2 = nn.Linear(hidden_features,   out_features)
            self.fc3 = nn.Linear(hidden_features,   out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # FC1 -> DW1
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_1(x, H, W)
        # FC2 -> DW2
        x = self.fc2(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_2(x, H, W)
        # FC3 -> DW3
        x = self.fc3(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_3(x, H, W)
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim)
        self.layer     = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim,
                                  act_layer=act_layer, drop=drop, no_kan=no_kan)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x

# ══════════════════════════════════════════════════════════════════════════════
#  Helper Modules (PatchEmbed, ConvLayer, D_ConvLayer)
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)
