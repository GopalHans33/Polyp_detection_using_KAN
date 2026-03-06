"""
metrics.py  —  KA-ResUNet++
=============================
All evaluation metrics: Dice, IoU, Precision, Recall, Specificity, HD95.

BUG FIXED:
  _to_binary_np was unconditionally calling torch.sigmoid(pred) on every input.
  train.py was passing binary {0, 1} tensors (already thresholded) into this function.
  sigmoid(0) = 0.5, sigmoid(1) = 0.731 — both above threshold 0.3 → EVERY pixel = positive.
  This caused Dice to freeze at the dataset's base polyp coverage rate (0.2440) every epoch.

  FIX: _to_binary_np no longer applies sigmoid. It expects inputs that are already
  probabilities in [0, 1]. train.py must pass probs (not pre-binarized preds).
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    from medpy.metric.binary import hd95 as medpy_hd95
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  AverageMeter
# ══════════════════════════════════════════════════════════════════════════════

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ══════════════════════════════════════════════════════════════════════════════
#  Core conversion helper
# ══════════════════════════════════════════════════════════════════════════════

def _to_binary_np(pred, target, threshold=0.5):
    """
    Convert pred and target to binary numpy uint8 arrays.

    IMPORTANT — CONTRACT FOR pred:
        pred must be a PROBABILITY tensor/array with values in [0, 1].
        Do NOT pass raw logits here. Apply sigmoid BEFORE calling any metric.
        Do NOT pass pre-binarized {0, 1} tensors here either — pass raw probs.

    WHY the old code was broken:
        The old version called torch.sigmoid(pred) unconditionally.
        train.py was passing preds = (probs > threshold).float() → values {0.0, 1.0}
        sigmoid(0.0) = 0.5, sigmoid(1.0) = 0.731
        With threshold=0.3: both 0.5 and 0.731 are > 0.3 → ALL pixels = positive
        → Dice = 2 * base_coverage / (1 + base_coverage) = 0.2440, frozen forever.
    """
    # Convert pred to numpy — NO sigmoid here, caller is responsible
    if torch.is_tensor(pred):
        pred_np = pred.detach().cpu().float().numpy()
    else:
        pred_np = np.array(pred, dtype=np.float32)

    # Convert target to numpy
    if torch.is_tensor(target):
        target_np = target.detach().cpu().float().numpy()
    else:
        target_np = np.array(target, dtype=np.float32)

    pred_bin   = (pred_np   > threshold).astype(np.uint8).flatten()
    target_bin = (target_np > 0.5      ).astype(np.uint8).flatten()

    return pred_bin, target_bin


# ══════════════════════════════════════════════════════════════════════════════
#  Core metric functions
# ══════════════════════════════════════════════════════════════════════════════

def dice_score(pred, target, threshold=0.5, smooth=1e-5):
    """Dice / F1 score. pred must be probability in [0,1]."""
    p, t = _to_binary_np(pred, target, threshold)
    inter = (p & t).sum()
    return float((2.0 * inter + smooth) / (p.sum() + t.sum() + smooth))


def iou_score(pred, target, threshold=0.5, smooth=1e-5):
    """Jaccard / IoU score. pred must be probability in [0,1]."""
    p, t = _to_binary_np(pred, target, threshold)
    inter = (p & t).sum()
    union = (p | t).sum()
    return float((inter + smooth) / (union + smooth))


def precision_score(pred, target, threshold=0.5, smooth=1e-5):
    """Precision = TP / (TP + FP). pred must be probability in [0,1]."""
    p, t = _to_binary_np(pred, target, threshold)
    tp = (p & t).sum()
    fp = (p.astype(bool) & ~t.astype(bool)).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def recall_score(pred, target, threshold=0.5, smooth=1e-5):
    """Recall = TP / (TP + FN). pred must be probability in [0,1]."""
    p, t = _to_binary_np(pred, target, threshold)
    tp = (p & t).sum()
    fn = (~p.astype(bool) & t.astype(bool)).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def specificity_score(pred, target, threshold=0.5, smooth=1e-5):
    """Specificity = TN / (TN + FP). pred must be probability in [0,1]."""
    p, t = _to_binary_np(pred, target, threshold)
    tn = (~p.astype(bool) & ~t.astype(bool)).sum()
    fp = ( p.astype(bool) & ~t.astype(bool)).sum()
    return float((tn + smooth) / (tn + fp + smooth))


def f1_score_metric(pred, target, threshold=0.5, smooth=1e-5):
    """F1 Score (harmonic mean of precision and recall). pred must be in [0,1]."""
    prec = precision_score(pred, target, threshold, smooth)
    rec  = recall_score(pred,   target, threshold, smooth)
    return float((2.0 * prec * rec + smooth) / (prec + rec + smooth))


def hd95_score(pred, target, threshold=0.5):
    """
    95th percentile Hausdorff Distance.
    Returns 0.0 if medpy missing or masks are empty.
    pred must be probability in [0,1].
    """
    if not MEDPY_AVAILABLE:
        return 0.0
    p, t = _to_binary_np(pred, target, threshold)
    if p.sum() == 0 or t.sum() == 0:
        return 0.0
    try:
        # Attempt to reshape to 2D for medpy
        n = len(p)
        side = int(n ** 0.5)
        p2d = p.reshape(side, side)
        t2d = t.reshape(side, side)
        return float(medpy_hd95(p2d, t2d))
    except Exception:
        return 0.0


def fpr_on_negatives(pred, target, threshold=0.5):
    """False Positive Rate on negative (polyp-free) samples."""
    p, t = _to_binary_np(pred, target, threshold)
    if t.sum() > 0:
        return None
    return float(p.sum()) / (len(p) + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
#  compute_all_metrics  —  expects pred to be probabilities in [0, 1]
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(pred, target, threshold=0.5):
    """
    Compute all metrics for a single image.

    Args:
        pred   : probability tensor or array in [0, 1] — NOT logits, NOT binary
        target : ground truth binary mask
        threshold: binarization threshold (default 0.5)
    """
    return {
        "dice":        dice_score(pred,        target, threshold),
        "iou":         iou_score(pred,         target, threshold),
        "precision":   precision_score(pred,   target, threshold),
        "recall":      recall_score(pred,      target, threshold),
        "specificity": specificity_score(pred, target, threshold),
        "f1":          f1_score_metric(pred,   target, threshold),
        "hd95":        hd95_score(pred,        target, threshold),
        "fpr":         fpr_on_negatives(pred,  target, threshold),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MetricsTracker
# ══════════════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """Tracks and averages metrics over an epoch."""
    METRIC_NAMES = ["dice", "iou", "precision", "recall", "specificity", "f1", "hd95"]

    def __init__(self):
        self.meters = {k: AverageMeter() for k in self.METRIC_NAMES}
        self.fpr_values = []

    def reset(self):
        for m in self.meters.values():
            m.reset()
        self.fpr_values = []

    def update(self, metrics_dict: dict, n: int = 1):
        for key in self.METRIC_NAMES:
            if key in metrics_dict and metrics_dict[key] is not None:
                self.meters[key].update(float(metrics_dict[key]), n)
        if metrics_dict.get("fpr") is not None:
            self.fpr_values.append(float(metrics_dict["fpr"]))

    def get_averages(self) -> dict:
        result = {k: self.meters[k].avg for k in self.METRIC_NAMES}
        if self.fpr_values:
            result["fpr_negatives"] = float(np.mean(self.fpr_values))
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  Size-stratified evaluation
# ══════════════════════════════════════════════════════════════════════════════

def compute_size_stratified_metrics(predictions, targets, threshold=0.5):
    """Compute Dice stratified by polyp coverage size."""
    bins = {"empty": [], "small": [], "medium": [], "large": [], "huge": []}

    if torch.is_tensor(predictions):
        predictions = predictions.cpu().detach()
    if torch.is_tensor(targets):
        targets = targets.cpu().detach()

    for i in range(len(predictions)):
        pred   = predictions[i]
        target = targets[i]
        t_np   = target.cpu().numpy() if torch.is_tensor(target) else np.array(target)
        coverage = (t_np > 0.5).sum() / t_np.size

        if   coverage == 0:       cat = "empty"
        elif coverage <= 0.05:    cat = "small"
        elif coverage <= 0.15:    cat = "medium"
        elif coverage <= 0.30:    cat = "large"
        else:                     cat = "huge"

        bins[cat].append(dice_score(pred, target, threshold))

    return {
        cat: {"dice": float(np.mean(s)) if s else 0.0, "count": len(s)}
        for cat, s in bins.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
#  InferenceTimer
# ══════════════════════════════════════════════════════════════════════════════

class InferenceTimer:
    def __init__(self):
        self.times = []
        self._t = 0.0

    def start(self):
        self._t = time.time()

    def stop(self):
        self.times.append(time.time() - self._t)

    def mean_ms(self):
        return float(np.mean(self.times)) * 1000 if self.times else 0.0
