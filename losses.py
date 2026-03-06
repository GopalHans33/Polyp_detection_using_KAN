"""
losses.py  —  KA-ResUNet++
============================
Custom loss functions: Dice, Boundary, and Combined Loss.
Matches parameters in config.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
#  1. Dice Loss (Standard)
# ══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Soft Dice Loss.
    Expects inputs to be probabilities (after sigmoid).
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten
        predictions = predictions.view(-1)
        targets     = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        union        = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


# ══════════════════════════════════════════════════════════════════════════════
#  2. Boundary Loss
# ══════════════════════════════════════════════════════════════════════════════

class BoundaryLoss(nn.Module):
    """
    BCE Loss specifically for the boundary (edge) prediction.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, boundary_logits: torch.Tensor, boundary_gt: torch.Tensor) -> torch.Tensor:
        return self.bce(boundary_logits, boundary_gt)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Combined Loss (The Master Function)
# ══════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    Composite loss function:
      L = Seg(BCE + Dice) + Boundary(BCE) + Aux4(BCE) + Aux3(BCE)
    """
    def __init__(
        self,
        pos_weight:  float = 2.0,
        bnd_weight:  float = 0.5,
        aux4_weight: float = 0.4,
        aux3_weight: float = 0.2,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        
        # Handle Positive Weight for Class Imbalance
        # We register it as a buffer so it automatically moves to GPU with the model
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        
        self.bce_seg  = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.bce_bnd  = nn.BCEWithLogitsLoss()
        self.bce_aux  = nn.BCEWithLogitsLoss()
        self.dice     = DiceLoss(smooth=1e-5)
        
        self.bnd_w    = bnd_weight
        self.aux4_w   = aux4_weight
        self.aux3_w   = aux3_weight
        self.dice_w   = dice_weight

    def forward(
        self,
        seg_logits:  torch.Tensor,   # Main Segmentation
        bnd_logits:  torch.Tensor,   # Boundary
        aux4_logits: torch.Tensor,   # Deep Sup 4
        aux3_logits: torch.Tensor,   # Deep Sup 3
        seg_gt:      torch.Tensor,   # Ground Truth Mask
        bnd_gt:      torch.Tensor,   # Ground Truth Boundary
    ):
        """
        Calculates weighted sum of all losses.
        """
        # 1. Main Segmentation Loss (BCE + Dice)
        l_bce_seg = self.bce_seg(seg_logits, seg_gt)
        l_dice    = self.dice(torch.sigmoid(seg_logits), seg_gt) * self.dice_w

        # 2. Boundary Loss
        l_bnd = self.bce_bnd(bnd_logits, bnd_gt) * self.bnd_w

        # 3. Deep Supervision (Auxiliary) Losses
        # We interpolate GT to match the size of auxiliary outputs if needed, 
        # or interpolate outputs to match GT. Standard is interpolate outputs to GT size.
        H, W = seg_gt.shape[2:]
        
        l_aux4 = 0.0
        if aux4_logits is not None:
            aux4_up = F.interpolate(aux4_logits, size=(H, W), mode='bilinear', align_corners=False)
            l_aux4 = self.bce_aux(aux4_up, seg_gt) * self.aux4_w
            
        l_aux3 = 0.0
        if aux3_logits is not None:
            aux3_up = F.interpolate(aux3_logits, size=(H, W), mode='bilinear', align_corners=False)
            l_aux3 = self.bce_aux(aux3_up, seg_gt) * self.aux3_w

        # Total Loss
        total = l_bce_seg + l_dice + l_bnd + l_aux4 + l_aux3

        loss_dict = {
            "total":   total.item(),
            "bce_seg": l_bce_seg.item(),
            "dice":    l_dice.item(),
            "bnd":     l_bnd.item(),
            "aux4":    l_aux4 if isinstance(l_aux4, float) else l_aux4.item(),
            "aux3":    l_aux3 if isinstance(l_aux3, float) else l_aux3.item(),
        }
        
        return total, loss_dict


# ── Factory ──────────────────────────────────────────────────────────────────

def build_criterion(cfg):
    """Builds the loss function using config weights."""
    return CombinedLoss(
        pos_weight  = cfg.POS_WEIGHT,
        bnd_weight  = cfg.BOUNDARY_WEIGHT,
        aux4_weight = cfg.AUX4_WEIGHT,
        aux3_weight = cfg.AUX3_WEIGHT,
        dice_weight = cfg.DICE_WEIGHT,
    ).to(cfg.DEVICE)
