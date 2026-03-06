"""
train.py  —  KA-ResUNet++
===========================
Training loop with Mixed Precision, Differential LR, and Logging.

BUG FIXED in validate_one_epoch:
  OLD (broken):
      probs = torch.sigmoid(seg_logits)
      preds = (probs > cfg.THRESHOLD).float()        ← pre-binarizes to {0.0, 1.0}
      metrics = compute_all_metrics(preds[i], ...)   ← passes binary tensor to metrics

  WHY IT WAS BROKEN:
      metrics.py._to_binary_np then called sigmoid(preds) on the already-binary tensor.
      sigmoid(0.0) = 0.5, sigmoid(1.0) = 0.731
      With cfg.THRESHOLD = 0.3: both 0.5 > 0.3 and 0.731 > 0.3 → ALL pixels = positive.
      This made Dice freeze at the dataset base coverage rate (0.2440) every single epoch.

  FIX (this version):
      probs = torch.sigmoid(seg_logits)               ← correct: logits → probs
      metrics = compute_all_metrics(probs[i], ...)    ← pass probs directly, NOT binary
      metrics.py now handles thresholding internally without applying sigmoid again.
"""

import os
import csv
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from metrics import MetricsTracker, compute_all_metrics, AverageMeter
from losses import build_criterion


# ══════════════════════════════════════════════════════════════════════════════
#  Optimizer  —  differential LR: KAN layers get lower LR
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model, cfg):
    kan_params   = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ['base_weight', 'spline_weight', 'spline_scaler', 'kan']):
            kan_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": kan_params,   "lr": cfg.KAN_LR,  "weight_decay": 0.0},
        {"params": other_params, "lr": cfg.LR,       "weight_decay": cfg.WEIGHT_DECAY},
    ]

    print(f"  [Optimizer] KAN params: {sum(p.numel() for p in kan_params):,} (LR={cfg.KAN_LR})")
    print(f"  [Optimizer] CNN params: {sum(p.numel() for p in other_params):,} (LR={cfg.LR})")

    return torch.optim.AdamW(param_groups)


# ══════════════════════════════════════════════════════════════════════════════
#  Scheduler
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.SCHEDULER_T0,
        T_mult=cfg.SCHEDULER_T_MULT,
        eta_min=cfg.SCHEDULER_ETA_MIN,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Train one epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler,
                    device, epoch, cfg):
    model.train()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    bnd_meter  = AverageMeter()

    autocast = torch.amp.autocast

    for batch_idx, (imgs, seg_gt, bnd_gt) in enumerate(loader):
        imgs   = imgs.to(device,   non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)
        bnd_gt = bnd_gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=cfg.MIXED_PRECISION):
            seg_logits, bnd_logits, aux4, aux3 = model(imgs)
            loss, loss_dict = criterion(
                seg_logits, bnd_logits, aux4, aux3, seg_gt, bnd_gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step(epoch + batch_idx / len(loader))

        n = imgs.size(0)
        loss_meter.update(loss_dict["total"], n)
        dice_meter.update(loss_dict.get("dice", 0), n)
        bnd_meter.update(loss_dict.get("bnd",  0), n)

        if (batch_idx + 1) % 10 == 0:
            lr_now = optimizer.param_groups[1]["lr"]
            print(
                f"  Epoch[{epoch+1}] Batch[{batch_idx+1}/{len(loader)}] "
                f"Loss={loss_meter.val:.4f} DiceLoss={dice_meter.val:.4f} LR={lr_now:.6f}"
            )

    return OrderedDict([
        ("train_loss", round(loss_meter.avg, 5)),
        ("train_dice", round(dice_meter.avg, 5)),
        ("train_bnd",  round(bnd_meter.avg,  5)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  Validate one epoch  ← THE BUG WAS HERE — NOW FIXED
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, cfg):
    """
    KEY FIX — two-line change that unblocks training:

    REMOVED this line:
        preds = (probs > cfg.THRESHOLD).float()   ← was pre-binarizing

    CHANGED this line:
        OLD: metrics = compute_all_metrics(preds[i], ...)   ← binary {0,1} input
        NEW: metrics = compute_all_metrics(probs[i], ...)   ← probability input

    metrics.py now receives probabilities in [0,1] and thresholds internally
    without applying sigmoid again. This gives correct Dice computation.
    """
    model.eval()

    loss_meter = AverageMeter()
    tracker    = MetricsTracker()
    tracker.reset()

    for imgs, seg_gt, bnd_gt in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)
        bnd_gt = bnd_gt.to(device, non_blocking=True)

        seg_logits, bnd_logits, aux4, aux3 = model(imgs)

        loss, _ = criterion(seg_logits, bnd_logits, aux4, aux3, seg_gt, bnd_gt)
        loss_meter.update(loss.item(), imgs.size(0))

        # Step 1: Convert logits → probabilities  [values in 0, 1]
        probs = torch.sigmoid(seg_logits)

        # Step 2: Pass probabilities directly — do NOT pre-binarize
        # metrics.py handles thresholding internally and no longer calls sigmoid
        for i in range(probs.size(0)):
            metrics = compute_all_metrics(
                probs[i].cpu(),      # ← probability tensor, NOT binary
                seg_gt[i].cpu(),
                cfg.THRESHOLD
            )
            tracker.update(metrics, n=1)

    avgs = tracker.get_averages()

    print(
        f"  Val Dice: {avgs['dice']:.4f} | IoU: {avgs['iou']:.4f} | "
        f"HD95: {avgs['hd95']:.2f}"
    )

    return OrderedDict([
        ("val_loss",        round(loss_meter.avg,         5)),
        ("val_dice",        round(avgs["dice"],            5)),
        ("val_iou",         round(avgs["iou"],             5)),
        ("val_precision",   round(avgs["precision"],       5)),
        ("val_recall",      round(avgs["recall"],          5)),
        ("val_specificity", round(avgs["specificity"],     5)),
        ("val_hd95",        round(avgs["hd95"],            3)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  CSV Logger
# ══════════════════════════════════════════════════════════════════════════════

class CSVLogger:
    def __init__(self, path):
        self.path    = path
        self.headers = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, metrics, epoch, lr):
        row = {"epoch": epoch + 1, "lr": round(lr, 8), **metrics}
        if self.headers is None:
            self.headers = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.headers).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.headers).writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
#  EarlyStopping
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = -1.0
        self.early_stop = False

    def __call__(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader, cfg):
    device = cfg.DEVICE
    cfg.make_dirs()

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.MIXED_PRECISION)
    stopper   = EarlyStopping(cfg.EARLY_STOP_PATIENCE)
    logger    = CSVLogger(cfg.LOG_CSV)

    best_dice = 0.0
    history   = []

    print("\n============================================================")
    print(f"Training Start: {cfg.PROJECT_NAME}")
    print(f"Device: {device} | Epochs: {cfg.NUM_EPOCHS} | Batch: {cfg.BATCH_SIZE}")
    print("============================================================\n")

    for epoch in range(cfg.NUM_EPOCHS):
        start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scaler, scheduler, device, epoch, cfg
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, cfg
        )

        lr_curr   = optimizer.param_groups[1]["lr"]
        val_dice  = val_metrics["val_dice"]
        epoch_sec = time.time() - start

        logger.write({**train_metrics, **val_metrics}, epoch, lr_curr)

        print(f"Epoch [{epoch+1:03d}/{cfg.NUM_EPOCHS}] Time: {epoch_sec:.0f}s | LR: {lr_curr:.6f}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Bnd: {train_metrics['train_bnd']:.4f}")
        print(f"  Val Dice: {val_dice:.4f} | IoU: {val_metrics['val_iou']:.4f} | HD95: {val_metrics['val_hd95']:.2f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch":         epoch + 1,
                "model_state":   model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "best_val_dice": best_dice,
            }, cfg.BEST_MODEL)
            print(f"★ New Best Model Saved ({best_dice:.4f})")

        history.append({**train_metrics, **val_metrics, "epoch": epoch + 1})

        if stopper(val_dice):
            print("\nEarly stopping triggered")
            break

        print("-" * 60)

    print(f"\nTraining Finished. Best Dice = {best_dice:.4f}")
    return best_dice, history
