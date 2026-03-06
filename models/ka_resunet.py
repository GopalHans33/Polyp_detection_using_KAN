"""
models/ka_resunet.py  —  KA-ResUNet++
========================================
Full Architecture: ResNet50 Encoder + KAN Bottleneck + Attention Decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ------------------------------------------------------------------------------
# FIX: Use Absolute Imports relative to the project root 'ka_resunet_plus'
# ------------------------------------------------------------------------------
try:
    from models.kan_modules import (
        KANBlock, PatchEmbed,
        ConvLayer, D_ConvLayer
    )
    from models.attention import AttentionGate, ChannelAttention
except ImportError:
    # Fallback: if running script directly inside 'models/' folder
    from kan_modules import (
        KANBlock, PatchEmbed,
        ConvLayer, D_ConvLayer
    )
    from attention import AttentionGate, ChannelAttention


class KAResUNet(nn.Module):
    """
    KA-ResUNet++ Architecture.
    
    Returns 4 outputs:
      1. seg_logits  (Main Segmentation)
      2. bnd_logits  (Boundary Prediction)
      3. aux4_logits (Deep Supervision Stage 4)
      4. aux3_logits (Deep Supervision Stage 3)
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
        print(f"  [Model] Loading ResNet50 (pretrained={pretrained})...")
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Encoder Stages
        self.enc0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        ) # -> H/4, 64
        self.enc1 = resnet.layer1  # -> H/4,  256
        self.enc2 = resnet.layer2  # -> H/8,  512
        self.enc3 = resnet.layer3  # -> H/16, 1024
        self.enc4 = resnet.layer4  # -> H/32, 2048

        # ── BOTTLENECK: KAN ───────────────────────────────────────────────────
        # B1: 2048 -> embed_dims[2]
        self.patch_b1 = PatchEmbed(
            img_size=8, patch_size=3, stride=2,
            in_chans=2048, embed_dim=embed_dims[2]
        )
        self.kan_b1 = KANBlock(
            dim=embed_dims[2], drop=drop_rate, drop_path=drop_path_rate
        )
        self.norm_b1 = nn.LayerNorm(embed_dims[2])

        # B2: embed_dims[2] -> embed_dims[1]
        self.patch_b2 = PatchEmbed(
            img_size=4, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[1]
        )
        self.kan_b2 = KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=drop_path_rate
        )
        self.norm_b2 = nn.LayerNorm(embed_dims[1])

        # ── ATTENTION GATES ───────────────────────────────────────────────────
        # Gate channels come from Decoder, Skip channels from Encoder
        self.ag4 = AttentionGate(F_g=embed_dims[1], F_l=2048, F_int=256)
        self.ag3 = AttentionGate(F_g=256,           F_l=1024, F_int=128)
        self.ag2 = AttentionGate(F_g=128,           F_l=512,  F_int=64)
        self.ag1 = AttentionGate(F_g=64,            F_l=256,  F_int=32)

        # ── DECODER ───────────────────────────────────────────────────────────
        # D_ConvLayer: Double Convolution Block
        self.dec4 = D_ConvLayer(embed_dims[1] + 2048, 256)
        self.dec3 = D_ConvLayer(256 + 1024, 128)
        self.dec2 = D_ConvLayer(128 + 512, 64)
        self.dec1 = D_ConvLayer(64 + 256, 32)
        
        # Final refinement
        self.dec0 = ConvLayer(32, 16)
        self.ca   = ChannelAttention(channels=16, reduction=4)

        # ── HEADS ─────────────────────────────────────────────────────────────
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.bnd_head = nn.Conv2d(16, 1, kernel_size=1)
        
        # Aux Heads for Deep Supervision
        self.aux4_head = nn.Conv2d(256, 1, kernel_size=1)
        self.aux3_head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        
        # Check input size (must be divisible by 32 for ResNet50)
        if H % 32 != 0 or W % 32 != 0:
            pass # Usually handled by dataset resizing

        # ── Encoder ───────────────────────────────────────────────────────────
        e0 = self.enc0(x)   # H/4
        e1 = self.enc1(e0)  # H/4
        e2 = self.enc2(e1)  # H/8
        e3 = self.enc3(e2)  # H/16
        e4 = self.enc4(e3)  # H/32

        # ── Bottleneck (KAN) ──────────────────────────────────────────────────
        # Patch Embed 1
        out, Hb, Wb = self.patch_b1(e4)
        out = self.kan_b1(out, Hb, Wb)
        out = self.norm_b1(out)
        out = out.reshape(B, Hb, Wb, -1).permute(0, 3, 1, 2).contiguous()

        # Patch Embed 2
        out, Hb2, Wb2 = self.patch_b2(out)
        out = self.kan_b2(out, Hb2, Wb2)
        out = self.norm_b2(out)
        out = out.reshape(B, Hb2, Wb2, -1).permute(0, 3, 1, 2).contiguous()

        # ── Decoder ───────────────────────────────────────────────────────────
        # Stage 4
        out = F.interpolate(out, size=e4.shape[2:], mode='bilinear', align_corners=False)
        e4_att = self.ag4(out, e4)
        d4 = self.dec4(torch.cat([out, e4_att], dim=1))
        aux4 = self.aux4_head(d4)

        # Stage 3
        d4_up = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        e3_att = self.ag3(d4_up, e3)
        d3 = self.dec3(torch.cat([d4_up, e3_att], dim=1))
        aux3 = self.aux3_head(d3)

        # Stage 2
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        e2_att = self.ag2(d3_up, e2)
        d2 = self.dec2(torch.cat([d3_up, e2_att], dim=1))

        # Stage 1
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        e1_att = self.ag1(d2_up, e1)
        d1 = self.dec1(torch.cat([d2_up, e1_att], dim=1))

        # Stage 0 (Output)
        d0 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=False)
        d0 = self.dec0(d0)
        d0 = self.ca(d0)

        # Heads
        seg_logits = self.seg_head(d0)
        bnd_logits = self.bnd_head(d0)

        return seg_logits, bnd_logits, aux4, aux3

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# Factory
def build_model(cfg, pretrained=True):
    return KAResUNet(
        num_classes=cfg.NUM_CLASSES,
        embed_dims=cfg.EMBED_DIMS,
        drop_rate=cfg.DROP_RATE,
        drop_path_rate=cfg.DROP_PATH_RATE,
        pretrained=pretrained
    )
