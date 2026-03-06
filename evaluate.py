"""
evaluate.py — KA-ResUNet++
Evaluation suite for Kvasir-SEG
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from metrics import (
    MetricsTracker,
    compute_all_metrics,
    compute_size_stratified_metrics,
    InferenceTimer
)

from inference import predict_tta


# ============================================================
# Core Evaluation Function
# ============================================================

@torch.no_grad()
def evaluate_on_loader(model, loader, cfg, use_tta=False, loader_name="test"):

    model.eval()
    device = cfg.DEVICE

    tracker = MetricsTracker()
    timer = InferenceTimer()

    all_preds = []
    all_gts = []

    print(f"\n[Evaluate] Running on {loader_name} (TTA={use_tta})")

    for batch in loader:

        # loader may return 2 or 3 items
        if len(batch) == 3:
            imgs, seg_gt, _ = batch
        else:
            imgs, seg_gt = batch

        imgs = imgs.to(device, non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)

        # =============================
        # Inference
        # =============================

        timer.start()

        if use_tta:
            prob = predict_tta(model, imgs, cfg.THRESHOLD)

        else:
            output = model(imgs)

            if isinstance(output, tuple):
                seg_logits = output[0]
            else:
                seg_logits = output

            prob = torch.sigmoid(seg_logits)

        timer.stop()

        # =============================
        # Convert to binary mask
        # =============================

        preds = (prob > cfg.THRESHOLD).float().cpu()
        gts = seg_gt.cpu()

        all_preds.append(preds)
        all_gts.append(gts)

        # =============================
        # Compute metrics
        # =============================

        for j in range(preds.size(0)):
            metrics = compute_all_metrics(preds[j], gts[j], cfg.THRESHOLD)
            tracker.update(metrics, n=1)

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    avgs = tracker.get_averages()
    avgs["inference_ms"] = timer.mean_ms()

    print(f"\n[{loader_name}] Results")
    print(
        f"Dice={avgs['dice']:.4f}  "
        f"IoU={avgs['iou']:.4f}  "
        f"Precision={avgs['precision']:.4f}  "
        f"Recall={avgs['recall']:.4f}"
    )

    print(
        f"Specificity={avgs['specificity']:.4f}  "
        f"F1={avgs['f1']:.4f}  "
        f"HD95={avgs['hd95']:.2f}  "
        f"Infer={avgs['inference_ms']:.2f}ms/img"
    )

    return avgs, all_preds, all_gts


# ============================================================
# Size Stratified Evaluation
# ============================================================

def evaluate_by_size(preds, gts, cfg):

    print("\n============================================================")
    print("SIZE STRATIFIED METRICS")
    print("============================================================")

    results = compute_size_stratified_metrics(preds, gts, cfg.THRESHOLD)

    print(f"{'Category':<12}{'Count':>8}{'Dice':>10}")
    print("-" * 30)

    for cat, vals in results.items():

        print(f"{cat:<12}{vals['count']:>8}{vals['dice']:>10.4f}")

    return results


# ============================================================
# Save Visualizations
# ============================================================

@torch.no_grad()
def save_visualizations(model, loader, cfg, n_samples=8, save_path=None):

    if save_path is None:
        save_path = os.path.join(cfg.RESULTS_DIR, "predictions.png")

    model.eval()
    device = cfg.DEVICE

    samples = []

    for batch in loader:

        if len(batch) == 3:
            imgs, seg_gt, bnd_gt = batch
        else:
            imgs, seg_gt = batch
            bnd_gt = torch.zeros_like(seg_gt)

        imgs_device = imgs.to(device)

        output = model(imgs_device)

        if isinstance(output, tuple):
            seg_logits = output[0]
            bnd_logits = output[1]
        else:
            seg_logits = output
            bnd_logits = torch.zeros_like(seg_logits)

        probs = torch.sigmoid(seg_logits).cpu()
        bnds = torch.sigmoid(bnd_logits).cpu()

        for i in range(imgs.size(0)):

            samples.append({
                "img": imgs[i].cpu(),
                "gt": seg_gt[i].cpu(),
                "pred": probs[i].cpu(),
                "bnd": bnds[i].cpu()
            })

            if len(samples) >= n_samples:
                break

        if len(samples) >= n_samples:
            break

    # =============================
    # Plot
    # =============================

    n = len(samples)

    fig = plt.figure(figsize=(12, n * 3))

    gs = gridspec.GridSpec(n, 4, figure=fig)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    titles = ["Input", "Ground Truth", "Prediction", "Boundary"]

    for row, s in enumerate(samples):

        img_np = s["img"].numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * std + mean, 0, 1)

        gt = s["gt"].squeeze().numpy()

        pred = (s["pred"].squeeze().numpy() > cfg.THRESHOLD).astype(np.float32)

        bnd = s["bnd"].squeeze().numpy()

        dice = compute_all_metrics(
            s["pred"],
            s["gt"],
            cfg.THRESHOLD
        )["dice"]

        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(img_np)
        ax.axis("off")
        if row == 0:
            ax.set_title(titles[0])

        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(gt, cmap="gray")
        ax.axis("off")
        if row == 0:
            ax.set_title(titles[1])

        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(pred, cmap="gray")
        ax.axis("off")
        if row == 0:
            ax.set_title(titles[2])

        ax.text(
            5,
            25,
            f"Dice: {dice:.3f}",
            color="white",
            fontsize=8,
            backgroundcolor="black"
        )

        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(bnd, cmap="hot")
        ax.axis("off")
        if row == 0:
            ax.set_title(titles[3])

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[Visualize] Saved predictions → {save_path}")


# ============================================================
# Full Evaluation Pipeline
# ============================================================

def run_full_evaluation(model, test_loader, cfg):

    print("\n############################################################")
    print(f"FULL EVALUATION — {cfg.PROJECT_NAME}")
    print("############################################################")

    results = {}

    metrics, preds, gts = evaluate_on_loader(
        model,
        test_loader,
        cfg,
        use_tta=cfg.USE_TTA,
        loader_name="Kvasir-SEG Test"
    )

    results["Kvasir"] = metrics

    size_results = evaluate_by_size(preds, gts, cfg)

    save_visualizations(model, test_loader, cfg)

    results_path = os.path.join(cfg.RESULTS_DIR, "final_results.txt")

    with open(results_path, "w") as f:

        f.write(f"{cfg.PROJECT_NAME} Final Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("Main Metrics\n")

        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\nSize Stratified Dice\n")

        for cat, vals in size_results.items():
            f.write(f"{cat}: {vals['dice']:.4f} (n={vals['count']})\n")

    print(f"\nResults saved → {results_path}")

    return results
