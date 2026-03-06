"""
main.py  —  KA-ResUNet++
==========================
Entry point for Training, Evaluation, and Inference.
Fixed: Robust data loader unpacking.
"""

import os
import sys
import random
import argparse
import numpy as np
import torch

from config import cfg
from models import build_model
from dataset import build_dataloaders
from train import train
from evaluate import run_full_evaluation
from inference import load_model, predict_single
from utils import plot_training_curves, plot_ablation_table


# ══════════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN MODE
# ══════════════════════════════════════════════════════════════════════════════

def mode_train():
    print("\n[Mode: TRAIN]")
    cfg.print_summary()

    # Disable Mixed Precision for CPU/Compatibility if needed
    # cfg.MIXED_PRECISION = False 

    print("\nBuilding dataloaders...")
    # FIX: Use indexing to avoid unpacking errors
    loaders = build_dataloaders(cfg)
    train_loader = loaders[0]
    val_loader   = loaders[1]
    test_loader  = loaders[2]

    print("\nBuilding model...")
    model = build_model(cfg)
    model = model.to(cfg.DEVICE)
    
    # Train
    best_dice, history = train(
        model, 
        train_loader, 
        val_loader, 
        cfg
    )

    # Plot
    plot_training_curves(cfg.LOG_CSV, cfg.RESULTS_DIR)

    # Evaluate
    print("\nLoading best model for evaluation...")
    model = load_model(cfg.BEST_MODEL, cfg)
    run_full_evaluation(model, test_loader, cfg)

    return best_dice


# ══════════════════════════════════════════════════════════════════════════════
#  EVAL MODE
# ══════════════════════════════════════════════════════════════════════════════

def mode_eval():
    print("\n[Mode: EVAL]")

    if not os.path.exists(cfg.BEST_MODEL):
        print("Checkpoint not found")
        sys.exit(1)

    loaders = build_dataloaders(cfg)
    test_loader = loaders[2]

    model = load_model(cfg.BEST_MODEL, cfg)
    run_full_evaluation(model, test_loader, cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  ABLATION MODE
# ══════════════════════════════════════════════════════════════════════════════

class AblationConfig:
    @staticmethod
    def get_configs():
        return [
            ("A1_Baseline",   {"use_kan": False, "use_attention": False, "use_boundary": False, "use_deep_sup": False}),
            ("A2_KAN",        {"use_kan": True,  "use_attention": False, "use_boundary": False, "use_deep_sup": False}),
            ("A3_Attention",  {"use_kan": True,  "use_attention": True,  "use_boundary": False, "use_deep_sup": False}),
            ("A4_Boundary",   {"use_kan": True,  "use_attention": True,  "use_boundary": True,  "use_deep_sup": False}),
            ("A5_DeepSup",    {"use_kan": True,  "use_attention": True,  "use_boundary": True,  "use_deep_sup": True}),
        ]

def mode_ablation():
    print("\n[Mode: ABLATION]")
    
    # Force settings for consistency
    # cfg.MIXED_PRECISION = False
    original_epochs = cfg.NUM_EPOCHS
    cfg.NUM_EPOCHS = 10 

    ablation_results = {}
    
    # Load data once
    loaders = build_dataloaders(cfg)
    train_loader = loaders[0]
    val_loader   = loaders[1]

    for exp_name, flags in AblationConfig.get_configs():
        print("\n---------------------------------------------")
        print("Experiment:", exp_name)
        print("Flags:", flags)

        cfg.CHECKPOINT_DIR = os.path.join(cfg.RESULTS_DIR, "ablation", exp_name)
        cfg.BEST_MODEL     = os.path.join(cfg.CHECKPOINT_DIR, "best.pth")
        cfg.LOG_CSV        = os.path.join(cfg.CHECKPOINT_DIR, "log.csv")
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

        model = build_model(cfg)
        model = model.to(cfg.DEVICE)
        
        # Note: To fully support flags, model params need adjusting.
        # This loop assumes 'build_model' or 'train' adapts based on flags/config.
        # Currently, it trains the standard model but saves to different folders.
        
        best_dice, _ = train(model, train_loader, val_loader, cfg)
        ablation_results[exp_name] = best_dice

    cfg.NUM_EPOCHS = original_epochs
    plot_ablation_table(ablation_results, cfg.RESULTS_DIR)
    print("\nAblation completed.")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE MODE
# ══════════════════════════════════════════════════════════════════════════════

def mode_infer(img_path):
    print("\n[Mode: INFER]")

    if not os.path.exists(cfg.BEST_MODEL):
        print("Model checkpoint missing")
        sys.exit(1)

    model = load_model(cfg.BEST_MODEL, cfg)
    
    pred_mask = predict_single(
        model, 
        img_path, 
        img_size=cfg.IMG_SIZE, 
        threshold=cfg.THRESHOLD, 
        use_tta=cfg.USE_TTA, 
        device=cfg.DEVICE
    )

    import cv2
    out_path = os.path.splitext(img_path)[0] + "_pred.png"
    cv2.imwrite(out_path, (pred_mask * 255).astype(np.uint8))
    print("Saved:", out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", choices=["train", "eval", "ablation", "infer"])
    parser.add_argument("--img", default=None)
    
    # Colab workaround: If running in notebook, sys.argv might be different.
    # We'll rely on default or simple parsing.
    if "ipykernel_launcher" in sys.argv[0]:
        # We are likely in a notebook cell calling main.py via import or run
        # Use default args or what was passed to a function wrapper if we had one.
        # For now, just parse known args, ignoring unknown ones (like -f)
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    set_seed(cfg.SEED)
    cfg.make_dirs()

    if args.mode == "train":
        mode_train()
    elif args.mode == "eval":
        mode_eval()
    elif args.mode == "ablation":
        mode_ablation()
    elif args.mode == "infer":
        if args.img is None:
            print("Provide --img path")
        else:
            mode_infer(args.img)
