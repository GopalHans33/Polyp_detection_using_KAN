"""
metrics.py  —  KA-ResUNet++
=============================
All evaluation metrics: Dice, IoU, Precision, Recall, Specificity, HD95.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

# Try importing medpy for HD95, but don't crash if missing
try:
    from medpy.metric.binary import hd95 as medpy_hd95
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False
    # No print here to keep logs clean during training


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
#  Core metric functions
# ══════════════════════════════════════════════════════════════════════════════

def _to_binary_np(pred, target, threshold=0.5):
    """Convert tensors or arrays to binary numpy."""
    if torch.is_tensor(pred):
        pred   = torch.sigmoid(pred).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    pred_bin   = (pred   > threshold).astype(np.uint8)
    target_bin = (target > 0.5      ).astype(np.uint8)
    return pred_bin, target_bin


def dice_score(pred, target, threshold=0.5, smooth=1e-5):
    """Dice / F1 score."""
    p, t = _to_binary_np(pred, target, threshold)
    inter = (p & t).sum()
    return (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def iou_score(pred, target, threshold=0.5, smooth=1e-5):
    """Jaccard / IoU score."""
    p, t = _to_binary_np(pred, target, threshold)
    inter = (p & t).sum()
    union = (p | t).sum()
    return (inter + smooth) / (union + smooth)


def precision_score(pred, target, threshold=0.5, smooth=1e-5):
    """Precision = TP / (TP + FP)."""
    p, t = _to_binary_np(pred, target, threshold)
    tp = (p & t).sum()
    fp = (p & ~t.astype(bool)).sum()
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred, target, threshold=0.5, smooth=1e-5):
    """Recall / Sensitivity = TP / (TP + FN)."""
    p, t = _to_binary_np(pred, target, threshold)
    tp = (p & t).sum()
    fn = (~p.astype(bool) & t.astype(bool)).sum()
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred, target, threshold=0.5, smooth=1e-5):
    """Specificity = TN / (TN + FP)."""
    p, t = _to_binary_np(pred, target, threshold)
    tn = (~p.astype(bool) & ~t.astype(bool)).sum()
    fp = ( p.astype(bool) & ~t.astype(bool)).sum()
    return (tn + smooth) / (tn + fp + smooth)


def f1_score_metric(pred, target, threshold=0.5, smooth=1e-5):
    """F1 Score (same as Dice for binary)."""
    prec = precision_score(pred, target, threshold, smooth)
    rec  = recall_score(pred,   target, threshold, smooth)
    return (2.0 * prec * rec + smooth) / (prec + rec + smooth)


def hd95_score(pred, target, threshold=0.5):
    """
    95th percentile Hausdorff Distance.
    Returns 0.0 if medpy is missing or masks are empty.
    """
    if not MEDPY_AVAILABLE:
        return 0.0
    
    p, t = _to_binary_np(pred, target, threshold)
    
    # HD95 requires both prediction and target to have some foreground
    if p.sum() == 0 or t.sum() == 0:
        return 0.0
        
    try:
        return float(medpy_hd95(p, t))
    except Exception:
        return 0.0


def fpr_on_negatives(pred, target, threshold=0.5):
    """False Positive Rate on negative (polyp-free) samples."""
    p, t = _to_binary_np(pred, target, threshold)
    # If target has polyp, this is not a negative sample
    if t.sum() > 0:
        return None  
        
    fp    = p.sum()
    total = p.size
    return float(fp) / (total + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
#  Full metrics dict
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(pred, target, threshold=0.5):
    """Compute all 9 metrics for a single image."""
    metrics = {
        "dice":        dice_score(pred,        target, threshold),
        "iou":         iou_score(pred,         target, threshold),
        "precision":   precision_score(pred,   target, threshold),
        "recall":      recall_score(pred,      target, threshold),
        "specificity": specificity_score(pred, target, threshold),
        "f1":          f1_score_metric(pred,   target, threshold),
        "hd95":        hd95_score(pred,        target, threshold),
        "fpr":         fpr_on_negatives(pred,  target, threshold),
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  MetricsTracker
# ══════════════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """Tracks metrics over an epoch."""
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
                self.meters[key].update(metrics_dict[key], n)
        
        if metrics_dict.get("fpr") is not None:
            self.fpr_values.append(metrics_dict["fpr"])

    def get_averages(self) -> dict:
        result = {k: self.meters[k].avg for k in self.METRIC_NAMES}
        if self.fpr_values:
            result["fpr_negatives"] = float(np.mean(self.fpr_values))
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  Size-stratified evaluation
# ══════════════════════════════════════════════════════════════════════════════

def compute_size_stratified_metrics(predictions, targets, threshold=0.5):
    """
    Compute Dice score stratified by polyp coverage size.
    """
    bins = {
        "empty":  [],   # coverage == 0
        "small":  [],   # 0 < coverage <= 0.05
        "medium": [],   # 0.05 < coverage <= 0.15
        "large":  [],   # 0.15 < coverage <= 0.30
        "huge":   [],   # coverage > 0.30
    }

    # If inputs are tensors, convert to list of numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().detach()
    if torch.is_tensor(targets):
        targets = targets.cpu().detach()

    for i in range(len(predictions)):
        pred = predictions[i]
        target = targets[i]
        
        if torch.is_tensor(target):
            t_np = target.cpu().numpy()
        else:
            t_np = np.array(target)
            
        coverage = (t_np > 0.5).sum() / t_np.size

        if coverage == 0:        cat = "empty"
        elif coverage <= 0.05:   cat = "small"
        elif coverage <= 0.15:   cat = "medium"
        elif coverage <= 0.30:   cat = "large"
        else:                    cat = "huge"

        score = dice_score(pred, target, threshold)
        bins[cat].append(score)

    results = {}
    for cat, scores in bins.items():
        results[cat] = {
            "dice":  float(np.mean(scores)) if scores else 0.0,
            "count": len(scores)
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Inference Timer
# ══════════════════════════════════════════════════════════════════════════════

class InferenceTimer:
    def __init__(self):
        self.times = []
        self._t = 0

    def start(self):
        self._t = time.time()

    def stop(self):
        self.times.append(time.time() - self._t)

    def mean_ms(self):
        return float(np.mean(self.times)) * 1000 if self.times else 0.0
