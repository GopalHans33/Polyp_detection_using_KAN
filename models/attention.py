"""
models/attention.py  —  KA-ResUNet++
======================================
Spatial Attention Gate module.
NEW — not in any of the 4 source notebooks.

Based on: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
Proven to add +2-4% Dice by suppressing irrelevant background in skip connections.

Architecture:
    g  : gating signal from decoder  [B, F_g, H, W]  — coarse, semantic
    x  : skip connection from encoder [B, F_l, H, W]  — fine, spatial
    out: x * alpha  where  alpha = sigmoid(W_psi( relu( W_g(g) + W_x(x) ) ))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Soft spatial attention gate for skip connections.

    Args:
        F_g   : channels of gating signal (from decoder)
        F_l   : channels of skip connection (from encoder)
        F_int : intermediate channels (typically F_l // 2)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # Transform gating signal to F_int channels
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Transform skip connection to F_int channels
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Collapse F_int → 1 to produce attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g : gating signal  [B, F_g, H_g, W_g]
            x : skip connection [B, F_l, H_x, W_x]
        Returns:
            x_attended : filtered skip connection [B, F_l, H_x, W_x]
        """
        # Upsample g to match x spatial size if they differ
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g1  = self.W_g(g)              # [B, F_int, H, W]
        x1  = self.W_x(x)              # [B, F_int, H, W]
        psi = self.relu(g1 + x1)       # element-wise add + ReLU
        psi = self.psi(psi)            # [B, 1, H, W]  — attention map in [0,1]

        return x * psi                 # broadcast multiply: filter skip connection


class ChannelAttention(nn.Module):
    """
    Lightweight channel attention (SE-style) — optional add-on for output head.
    Learns which feature channels to emphasise before final prediction.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)
