"""
models/kan_modules.py  —  KA-ResUNet++
========================================
Complete KAN stack taken from Code 2 (train_khaolun3_code2.ipynb).
Includes: KANLinear, KANLayer, KANBlock, FastKANConvLayer,
          all basis functions, helper Conv modules, PatchEmbed.

CHANGES from Code 2:
  - None to KAN logic (it was already correct)
  - Added module docstrings
  - Cleaned unused imports
"""

import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ══════════════════════════════════════════════════════════════════════════════
#  KANLinear  —  B-spline based learnable activation on edges
# ══════════════════════════════════════════════════════════════════════════════

class KANLinear(torch.nn.Module):
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
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise   = scale_noise
        self.scale_base    = scale_base
        self.scale_spline  = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps      = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
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
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """Compute B-spline bases. x: (batch, in_features)"""
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
        """Fit spline coefficients. x:(batch,in), y:(batch,in,out)"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
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

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step + x_sorted[0] - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate([
            grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
        ], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Basis Function Variants
# ══════════════════════════════════════════════════════════════════════════════

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min=-2., grid_max=2., num_grids=4, denominator=None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class BSplineFunction(nn.Module):
    def __init__(self, grid_min=-2., grid_max=2., degree=3, num_basis=8):
        super(BSplineFunction, self).__init__()
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
        basis_functions = torch.stack(
            [self.basis_function(i, self.degree, x) for i in range(self.num_basis)], dim=-1
        )
        return basis_functions


class ChebyshevFunction(nn.Module):
    def __init__(self, degree=4):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        chebyshev = [torch.ones_like(x), x]
        for n in range(2, self.degree):
            chebyshev.append(2 * x * chebyshev[-1] - chebyshev[-2])
        return torch.stack(chebyshev, dim=-1)


class FourierBasisFunction(nn.Module):
    def __init__(self, num_frequencies=4, period=1.0):
        super().__init__()
        assert num_frequencies % 2 == 0
        self.num_frequencies = num_frequencies
        self.period = nn.Parameter(torch.Tensor([period]), requires_grad=False)

    def forward(self, x):
        freqs = torch.arange(1, self.num_frequencies // 2 + 1, device=x.device)
        sin_c = torch.sin(2 * torch.pi * freqs * x[..., None] / self.period)
        cos_c = torch.cos(2 * torch.pi * freqs * x[..., None] / self.period)
        return torch.cat([sin_c, cos_c], dim=-1)


class PolynomialFunction(nn.Module):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        return torch.stack([x ** i for i in range(self.degree)], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  SplineConv2D  +  FastKANConvLayer
# ══════════════════════════════════════════════════════════════════════════════

class SplineConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, init_scale=0.1,
                 padding_mode="zeros", **kw):
        self.init_scale = init_scale
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode, **kw)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FastKANConvLayer(nn.Module):
    """Spatial KAN convolution — applies basis function expansion then convolves."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, grid_min=-2., grid_max=2.,
                 num_grids=4, use_base_update=True, base_activation=F.silu,
                 spline_weight_init_scale=0.1, padding_mode="zeros",
                 kan_type="BSpline"):
        super().__init__()
        if kan_type == "RBF":
            self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        elif kan_type == "Fourier":
            self.rbf = FourierBasisFunction(num_grids)
        elif kan_type == "Poly":
            self.rbf = PolynomialFunction(num_grids)
        elif kan_type == "Chebyshev":
            self.rbf = ChebyshevFunction(num_grids)
        else:  # BSpline (default)
            self.rbf = BSplineFunction(grid_min, grid_max, 4, num_grids)

        self.spline_conv = SplineConv2D(
            in_channels * num_grids, out_channels, kernel_size,
            stride, padding, dilation, groups, bias,
            spline_weight_init_scale, padding_mode
        )
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias, padding_mode)

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
#  Depthwise Helpers
# ══════════════════════════════════════════════════════════════════════════════

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


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


# ══════════════════════════════════════════════════════════════════════════════
#  KANLayer  —  3 KANLinear + 3 DW_bn_relu interleaved
# ══════════════════════════════════════════════════════════════════════════════

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
            base_activation=torch.nn.SiLU,
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dwconv_3(x, H, W)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  KANBlock  —  Pre-norm + DropPath + KANLayer
# ══════════════════════════════════════════════════════════════════════════════

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim)
        self.layer     = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim,
                                  act_layer=act_layer, drop=drop, no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  PatchEmbed  —  Overlapping patch tokenization
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size   = img_size
        self.patch_size = patch_size
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ══════════════════════════════════════════════════════════════════════════════
#  Conv Block Helpers  (used in encoder/decoder)
# ══════════════════════════════════════════════════════════════════════════════

class KConvLayer(nn.Module):
    """Double FastKAN conv block — used as KAN-enhanced encoder stage."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = FastKANConvLayer(in_ch,  out_ch, 3, padding=1,
                                      kan_type="BSpline", num_grids=4,
                                      grid_min=-2., grid_max=2.)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = FastKANConvLayer(out_ch, out_ch, 3, padding=1,
                                      kan_type="BSpline", num_grids=4,
                                      grid_min=-2., grid_max=2.)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn2(self.conv2(self.bn1(self.conv1(x))))


class ConvLayer(nn.Module):
    """Standard double conv block."""
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

    def forward(self, x):
        return self.conv(x)


class D_ConvLayer(nn.Module):
    """Decoder double conv: in→in→out (from Code 2)."""
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

    def forward(self, x):
        return self.conv(x)
