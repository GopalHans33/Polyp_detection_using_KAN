"""
inference.py  —  KA-ResUNet++
================================
Single-image inference + Test-Time Augmentation (TTA).
Includes bug fixes for transpose augmentation.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: Safe Model Call
# ══════════════════════════════════════════════════════════════════════════════

def _get_seg_logits(model, x):
    """
    Safely extract segmentation logits from model output.
    Handles cases where model returns a tuple (seg, bnd, ...) or just seg.
    """
    output = model(x)
    if isinstance(output, tuple):
        return output[0]
    return output

# ══════════════════════════════════════════════════════════════════════════════
#  predict_tta  (Test Time Augmentation)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_tta(model, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Test-Time Augmentation: average 4 forward passes.
        Pass 1: original
        Pass 2: horizontal flip  → flip output back
        Pass 3: vertical flip    → flip output back
        Pass 4: transpose        → transpose output back
    """
    model.eval()
    outs = []

    # Pass 1: Original
    logits = _get_seg_logits(model, x)
    outs.append(torch.sigmoid(logits))

    # Pass 2: Horizontal Flip
    x_hf = torch.flip(x, dims=[3])
    logits_hf = _get_seg_logits(model, x_hf)
    outs.append(torch.flip(torch.sigmoid(logits_hf), dims=[3]))

    # Pass 3: Vertical Flip
    x_vf = torch.flip(x, dims=[2])
    logits_vf = _get_seg_logits(model, x_vf)
    outs.append(torch.flip(torch.sigmoid(logits_vf), dims=[2]))

    # Pass 4: Transpose (swap H and W)
    # Note: contiguous() is needed after transpose in memory
    x_t = x.transpose(2, 3)
    logits_t = _get_seg_logits(model, x_t)
    # Transpose OUTPUT back to original shape
    outs.append(torch.sigmoid(logits_t).transpose(2, 3)) 

    # Average predictions
    prob_avg = torch.stack(outs, dim=0).mean(dim=0)
    return prob_avg


# ══════════════════════════════════════════════════════════════════════════════
#  predict_single  —  Inference on file path
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single(
    model,
    img_path: str,
    img_size: int = 256,
    threshold: float = 0.5,
    use_tta: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run inference on a single image file.
    Returns: binary numpy mask [H, W] (0 or 1)
    """
    # ── Load & Preprocess ──────────────────────────────────────────────────
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # Resize
    img_resized = cv2.resize(image, (img_size, img_size))
    
    # Normalize (ImageNet stats - MUST match dataset.py)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    
    # To Tensor [B, C, H, W]
    x = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────
    model.eval()
    if use_tta:
        prob = predict_tta(model, x, threshold)
    else:
        logits = _get_seg_logits(model, x)
        prob = torch.sigmoid(logits)

    # ── Postprocess ───────────────────────────────────────────────────────
    prob_np = prob.squeeze().cpu().numpy()
    pred_mask = (prob_np > threshold).astype(np.uint8)

    # Resize back to original resolution if needed
    if (orig_h, orig_w) != (img_size, img_size):
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
    return pred_mask


# ══════════════════════════════════════════════════════════════════════════════
#  overlay_prediction  —  Visualization Helper
# ══════════════════════════════════════════════════════════════════════════════

def overlay_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay prediction (Green) and GT (Red) on image.
    Overlap (TP) becomes Yellow.
    """
    overlay = image.copy().astype(np.float32)
    h, w    = image.shape[:2]

    # Resize masks to match image
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_mask is not None and gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_bool = pred_mask.astype(bool)

    if gt_mask is not None:
        gt_bool = gt_mask.astype(bool)
        
        # Colors
        green = np.array([0, 255, 0])
        red   = np.array([255, 0, 0])
        
        # TP (True Positive) -> Green (or Yellowish if we mix red+green)
        tp = pred_bool & gt_bool
        # FP (False Positive) -> Blue (Prediction but no GT) - let's stick to Green for Pred
        fp = pred_bool & ~gt_bool
        # FN (False Negative) -> Red (GT but no Pred)
        fn = ~pred_bool & gt_bool

        # Standard Medical convention: 
        # GT = Green outline? Or Red fill? 
        # Let's do: Pred = Green, GT = Red.
        # Intersection will look Yellow.
        
        overlay[gt_bool]   = overlay[gt_bool] * (1 - alpha) + red * alpha
        overlay[pred_bool] = overlay[pred_bool] * (1 - alpha) + green * alpha
        
    else:
        # Just prediction
        green = np.array([0, 255, 0])
        overlay[pred_bool] = overlay[pred_bool] * (1 - alpha) + green * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  load_model
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, cfg):
    """Load model architecture and weights for inference."""
    # Local import to avoid circular dependency
    from models import build_model
    
    print(f"  [Inference] Loading model architecture...")
    model = build_model(cfg) 
    
    print(f"  [Inference] Loading weights from: {checkpoint_path}")
    if torch.cuda.is_available():
        ckpt = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    else:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
    # Handle state dict (remove 'module.' prefix if saved from DataParallel)
    state_dict = ckpt["model_state"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(cfg.DEVICE)
    model.eval()
    
    if "best_val_dice" in ckpt:
        print(f"  [Inference] Checkpoint Val Dice: {ckpt['best_val_dice']:.4f}")
        
    return model
