"""
models/ka_resunet.py  —  KA-ResUNet++
========================================
Full KA-ResUNet++ architecture.

Design:
  Encoder   : ResNet50 pretrained on ImageNet (5 skip connections)
  Bottleneck: 2× KANBlock (tokenized KAN processing)
  Decoder   : 4 stages, each with AttentionGate on skip connection
  Heads     : segmentation + boundary (dual output) + 2 aux heads (deep supervision)

Returns (seg_logits, bnd_logits, aux4_logits, aux3_logits)
  - All are raw logits (no sigmoid) — use BCEWithLogitsLoss
  - At inference: apply sigmoid to seg_logits, threshold at 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import trunc_normal_

from .kan_modules import (
    KANBlock, PatchEmbed,
    ConvLayer, D_ConvLayer, KConvLayer
)
from .attention import AttentionGate, ChannelAttention


class KAResUNet(nn.Module):
    """
    KA-ResUNet++ : Kolmogorov-Arnold Attention ResNet U-Net++

    Args:
        num_classes    : number of output channels (1 for binary segmentation)
        embed_dims     : [d_dec4, d_dec3, d_bottleneck] for KAN stages
        drop_rate      : dropout in KANLayer
        drop_path_rate : stochastic depth in KANBlock
        pretrained     : load ImageNet weights for ResNet50 encoder
    """

    def __init__(
        self,
        num_classes: int = 1,
        embed_dims: list = [128, 160, 256],
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── ENCODER: ResNet50 ─────────────────────────────────────────────────
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Stem: conv1 + bn1 + relu + maxpool  →  64 ch, H/4
        self.enc0 = nn.Sequential(
            resnet.conv1,   # 3→64, stride=2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # stride=2  → total stride=4
        )
        self.enc1 = resnet.layer1   # 64  → 256  ch, stride=1  (H/4)
        self.enc2 = resnet.layer2   # 256 → 512  ch, stride=2  (H/8)
        self.enc3 = resnet.layer3   # 512 → 1024 ch, stride=2  (H/16)
        self.enc4 = resnet.layer4   # 1024→ 2048 ch, stride=2  (H/32)

        # Channel counts at each encoder stage
        # enc0: 64   enc1: 256   enc2: 512   enc3: 1024   enc4: 2048

        # ── BOTTLENECK: 2× KAN ────────────────────────────────────────────────
        # Stage B1: 2048 → embed_dims[2]  (e.g. 256)
        self.patch_b1 = PatchEmbed(
            img_size=8, patch_size=3, stride=2,
            in_chans=2048, embed_dim=embed_dims[2]
        )
        self.kan_b1 = KANBlock(
            dim=embed_dims[2], drop=drop_rate, drop_path=drop_path_rate
        )
        self.norm_b1 = nn.LayerNorm(embed_dims[2])

        # Stage B2: embed_dims[2] → embed_dims[1]  (e.g. 160)
        self.patch_b2 = PatchEmbed(
            img_size=4, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[1]
        )
        self.kan_b2 = KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=drop_path_rate
        )
        self.norm_b2 = nn.LayerNorm(embed_dims[1])

        # ── ATTENTION GATES ───────────────────────────────────────────────────
        # AG4: gate=embed_dims[1], skip=enc4=2048
        self.ag4 = AttentionGate(F_g=embed_dims[1], F_l=2048, F_int=256)
        # AG3: gate=dec4_out, skip=enc3=1024
        self.ag3 = AttentionGate(F_g=256, F_l=1024, F_int=128)
        # AG2: gate=dec3_out, skip=enc2=512
        self.ag2 = AttentionGate(F_g=128, F_l=512, F_int=64)
        # AG1: gate=dec2_out, skip=enc1=256
        self.ag1 = AttentionGate(F_g=64, F_l=256, F_int=32)

        # ── DECODER ───────────────────────────────────────────────────────────
        # D4: cat(embed_dims[1], 2048) → 256
        self.dec4 = D_ConvLayer(embed_dims[1] + 2048, 256)
        # D3: cat(256, 1024) → 128
        self.dec3 = D_ConvLayer(256 + 1024, 128)
        # D2: cat(128, 512) → 64
        self.dec2 = D_ConvLayer(128 + 512, 64)
        # D1: cat(64, 256) → 32
        self.dec1 = D_ConvLayer(64 + 256, 32)
        # D0: 32 → 16  (no skip here — just refine)
        self.dec0 = ConvLayer(32, 16)

        # Optional channel attention on final decoder features
        self.ca = ChannelAttention(channels=16, reduction=4)

        # ── OUTPUT HEADS ──────────────────────────────────────────────────────
        # Primary segmentation head
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)

        # Boundary head (separate branch — supervised with boundary GT)
        self.bnd_head = nn.Conv2d(16, 1, kernel_size=1)

        # Deep supervision heads (auxiliary losses at intermediate decoder stages)
        self.aux4_head = nn.Conv2d(256, 1, kernel_size=1)   # at dec4 output
        self.aux3_head = nn.Conv2d(128, 1, kernel_size=1)   # at dec3 output

        # ── WEIGHT INIT for new layers ─────────────────────────────────────────
        self._init_new_layers()

    def _init_new_layers(self):
        """Initialize all layers except the pretrained ResNet encoder."""
        modules_to_init = [
            self.ag4, self.ag3, self.ag2, self.ag1,
            self.dec4, self.dec3, self.dec2, self.dec1, self.dec0,
            self.ca, self.seg_head, self.bnd_head,
            self.aux4_head, self.aux3_head,
        ]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def freeze_encoder(self, freeze: bool = True):
        """Freeze/unfreeze ResNet encoder weights."""
        for param in self.enc0.parameters(): param.requires_grad = not freeze
        for param in self.enc1.parameters(): param.requires_grad = not freeze
        for param in self.enc2.parameters(): param.requires_grad = not freeze
        for param in self.enc3.parameters(): param.requires_grad = not freeze
        for param in self.enc4.parameters(): param.requires_grad = not freeze

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : input image [B, 3, H, W]

        Returns:
            seg_logits  : [B, 1, H, W]  — segmentation logits
            bnd_logits  : [B, 1, H, W]  — boundary logits
            aux4_logits : [B, 1, H/8, W/8]  — deep supervision stage 4
            aux3_logits : [B, 1, H/16, W/16] — deep supervision stage 3
        """
        B, _, H, W = x.shape

        # ── ENCODER ───────────────────────────────────────────────────────────
        e0 = self.enc0(x)    # [B,   64, H/4,  W/4 ]
        e1 = self.enc1(e0)   # [B,  256, H/4,  W/4 ]
        e2 = self.enc2(e1)   # [B,  512, H/8,  W/8 ]
        e3 = self.enc3(e2)   # [B, 1024, H/16, W/16]
        e4 = self.enc4(e3)   # [B, 2048, H/32, W/32]

        # ── KAN BOTTLENECK ────────────────────────────────────────────────────
        # B1: 2048 → embed_dims[2]
        out, Hb, Wb = self.patch_b1(e4)        # tokenize
        out = self.kan_b1(out, Hb, Wb)          # KAN processing
        out = self.norm_b1(out)
        out = out.reshape(B, Hb, Wb, -1).permute(0, 3, 1, 2).contiguous()
        # [B, embed_dims[2], ~H/64, ~W/64]

        # B2: embed_dims[2] → embed_dims[1]
        out, Hb2, Wb2 = self.patch_b2(out)
        out = self.kan_b2(out, Hb2, Wb2)
        out = self.norm_b2(out)
        out = out.reshape(B, Hb2, Wb2, -1).permute(0, 3, 1, 2).contiguous()
        # [B, embed_dims[1], ~H/128, ~W/128]

        # ── DECODER + ATTENTION GATES ─────────────────────────────────────────
        # Stage 4
        out = F.interpolate(out, size=e4.shape[2:], mode='bilinear', align_corners=False)
        e4_att = self.ag4(out, e4)                          # attend skip
        d4 = self.dec4(torch.cat([out, e4_att], dim=1))     # [B, 256, H/32, W/32]
        aux4_logits = self.aux4_head(d4)                    # deep supervision

        # Stage 3
        d4_up = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        e3_att = self.ag3(d4_up, e3)
        d3 = self.dec3(torch.cat([d4_up, e3_att], dim=1))   # [B, 128, H/16, W/16]
        aux3_logits = self.aux3_head(d3)                    # deep supervision

        # Stage 2
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        e2_att = self.ag2(d3_up, e2)
        d2 = self.dec2(torch.cat([d3_up, e2_att], dim=1))   # [B, 64, H/8, W/8]

        # Stage 1
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        e1_att = self.ag1(d2_up, e1)
        d1 = self.dec1(torch.cat([d2_up, e1_att], dim=1))   # [B, 32, H/4, W/4]

        # Stage 0: upsample to full resolution
        d0 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=False)
        d0 = self.dec0(d0)                                  # [B, 16, H, W]
        d0 = self.ca(d0)                                    # channel attention

        # ── OUTPUT HEADS ──────────────────────────────────────────────────────
        seg_logits = self.seg_head(d0)   # [B, 1, H, W]
        bnd_logits = self.bnd_head(d0)   # [B, 1, H, W]

        return seg_logits, bnd_logits, aux4_logits, aux3_logits

    def get_param_groups(self, lr: float, kan_lr: float, weight_decay: float):
        """
        Returns param groups with differential learning rates:
          - KAN fc layers → kan_lr  (lower — spline weights need gentle updates)
          - Everything else → lr
        """
        kan_params   = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'layer' in name.lower() and 'fc' in name.lower():
                kan_params.append(param)
            else:
                other_params.append(param)
        return [
            {'params': kan_params,   'lr': kan_lr,  'weight_decay': 0.0},
            {'params': other_params, 'lr': lr,       'weight_decay': weight_decay},
        ]

    def count_parameters(self):
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ── Quick model factory ──────────────────────────────────────────────────────

def build_model(cfg, pretrained: bool = True) -> KAResUNet:
    model = KAResUNet(
        num_classes    = cfg.NUM_CLASSES,
        embed_dims     = cfg.EMBED_DIMS,
        drop_rate      = cfg.DROP_RATE,
        drop_path_rate = cfg.DROP_PATH_RATE,
        pretrained     = pretrained,
    )
    return model.to(cfg.DEVICE)
