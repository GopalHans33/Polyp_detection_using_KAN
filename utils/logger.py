"""
utils/logger.py  —  KA-ResUNet++
Training curve visualization.
"""

import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_curves(log_csv: str, save_dir: str):
    """
    Plot training and validation curves from CSV log.
    Generates 3 plots: Loss, Dice/IoU, HD95.
    """
    if not os.path.exists(log_csv):
        print(f"[Logger] Log file not found: {log_csv}")
        return

    # Read CSV
    try:
        df = pd.read_csv(log_csv)
    except Exception as e:
        print(f"[Logger] Error reading CSV: {e}")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 1. Loss Curve ────────────────────────────────────────────────────────
    if "train_loss" in df.columns and "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="steelblue")
        axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="coral")
        axes[0].set_title("Loss", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    else:
        axes[0].set_title("Loss (Data Missing)")
        axes[0].text(0.5, 0.5, "No Loss Data", ha='center')

    # ── 2. Dice / IoU ────────────────────────────────────────────────────────
    # Check for available metrics
    has_dice = "val_dice" in df.columns
    has_iou  = "val_iou" in df.columns
    
    if has_dice:
        # Plot Train Dice if available (some loggers only do val)
        if "train_dice" in df.columns:
            axes[1].plot(df["epoch"], df["train_dice"], label="Train Dice", color="steelblue", linestyle="--")
        
        axes[1].plot(df["epoch"], df["val_dice"],   label="Val Dice",   color="coral")
    
    if has_iou:
        axes[1].plot(df["epoch"], df["val_iou"],    label="Val IoU",    color="green")

    axes[1].set_title("Segmentation Metrics", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (0-1)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # ── 3. HD95 (Hausdorff Distance) ─────────────────────────────────────────
    if "val_hd95" in df.columns:
        axes[2].plot(df["epoch"], df["val_hd95"], label="Val HD95", color="purple")
        axes[2].set_title("Hausdorff Distance 95", fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("HD95 (pixels)")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    else:
        # Hide the third plot if no HD95 data (common in early testing)
        axes[2].axis('off')

    plt.suptitle("KA-ResUNet++ Training Progress", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    
    # Save
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Logger] Curves saved to: {out_path}")


def plot_ablation_table(results: dict, save_dir: str):
    """
    Bar chart of Dice scores across ablation experiments.
    results: {'Baseline': 0.880, '+KAN': 0.893, ...}
    """
    if not results:
        return

    labels = list(results.keys())
    scores = list(results.values())
    
    # Create colors based on score magnitude
    min_s, max_s = min(scores), max(scores)
    denom = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0
    
    colors = plt.cm.Blues([(s - min_s) / denom * 0.6 + 0.4 for s in scores])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor="navy", linewidth=0.8)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    
    # Zoom in on the relevant Y-axis range
    ax.set_ylim(max(0.0, min_s - 0.05), min(1.0, max_s + 0.05))
    
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Ablation Study Results", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add text labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "ablation_chart.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Logger] Ablation chart saved to: {out_path}")
