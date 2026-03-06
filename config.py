"""

config.py — KA-ResUNet++

Single source of truth for ALL hyperparameters.

Updated for: Local/Server Environment (Dynamic Pathing)

"""

import os

import torch



class Config:

    # ── GENERAL ──────────────────────────────────────────────────────────────

    PROJECT_NAME        = "KA-ResUNet++"

    SEED                = 42

    DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

    # Adjust workers based on your CPU. 2-4 is usually safe for local machines.

    NUM_WORKERS         = 2

    PIN_MEMORY          = True



    # ── IMAGE ─────────────────────────────────────────────────────────────────

    IMG_SIZE            = 256

    IN_CHANNELS         = 3



    # ── MODEL ─────────────────────────────────────────────────────────────────

    NUM_CLASSES         = 1

    EMBED_DIMS          = [128, 160, 256]

    DROP_RATE           = 0.1

    DROP_PATH_RATE      = 0.1

    KAN_GRID_SIZE       = 5

    KAN_SPLINE_ORDER    = 3

    KAN_NUM_GRIDS       = 4

    KAN_GRID_MIN        = -2.0

    KAN_GRID_MAX        = 2.0



    # ── TRAINING ──────────────────────────────────────────────────────────────

    BATCH_SIZE          = 8          # Reduce to 4 or 2 if you hit CUDA OOM

    VAL_BATCH_SIZE      = 4

    NUM_EPOCHS          = 100

    LR                  = 1e-4

    KAN_LR              = 5e-5

    WEIGHT_DECAY        = 1e-4

    GRAD_CLIP           = 1.0

    EARLY_STOP_PATIENCE = 15

    MIXED_PRECISION     = True



    # Scheduler

    SCHEDULER_T0        = 10

    SCHEDULER_T_MULT    = 2

    SCHEDULER_ETA_MIN   = 1e-6



    # ── LOSS WEIGHTS ──────────────────────────────────────────────────────────

    POS_WEIGHT          = 2.0

    BOUNDARY_WEIGHT     = 0.5

    AUX4_WEIGHT         = 0.4

    AUX3_WEIGHT         = 0.2

    DICE_WEIGHT         = 1.0



    # ── METRICS ───────────────────────────────────────────────────────────────

    THRESHOLD           = 0.5

    SMOOTH              = 1e-5



    # ── TTA (Test Time Augmentation) ──────────────────────────────────────────

    USE_TTA             = True



    # ── DATASET PATHS (DYNAMIC) ───────────────────────────────────────────────

    # We automatically determine the project root based on this file's location.

    # Structure:

    #   ka_resunet_plus/

    #       config.py

    #       data/

    #           Kvasir/

    #               images/

    #               masks/

   

    # Get the directory where config.py is currently located

    PROJECT_ROOT        = os.path.dirname(os.path.abspath(__file__))

   

    # Point to the data directory shown in your screenshot

    DATASET_ROOT        = os.path.join(PROJECT_ROOT, "data", "Kvasir")



    # These are built automatically

    KVASIR_IMG_DIR      = os.path.join(DATASET_ROOT, "images")

    KVASIR_MASK_DIR     = os.path.join(DATASET_ROOT, "masks")



    # ── SPLITS (80% train / 10% val / 10% test) ──────────────────────────────

    TRAIN_RATIO         = 0.80

    VAL_RATIO           = 0.10

    TEST_RATIO          = 0.10



    # ── OUTPUT PATHS ──────────────────────────────────────────────────────────

    # Save results to the local folders 'checkpoints' and 'results'

    # found in your directory tree.

    CHECKPOINT_DIR      = os.path.join(PROJECT_ROOT, "checkpoints")

    RESULTS_DIR         = os.path.join(PROJECT_ROOT, "results")

   

    BEST_MODEL          = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    LOG_CSV             = os.path.join(RESULTS_DIR, "training_log.csv")



    BOUNDARY_KERNEL_SIZE = 5



    @classmethod

    def make_dirs(cls):

        """Creates the checkpoint and results directories if they don't exist."""

        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)

        os.makedirs(cls.RESULTS_DIR,    exist_ok=True)



    @classmethod

    def print_summary(cls):

        print("=" * 60)

        print(f"  {cls.PROJECT_NAME}  —  Configuration")

        print("=" * 60)

        print(f"  Device         : {cls.DEVICE}")

        print(f"  Project Root   : {cls.PROJECT_ROOT}")

        print(f"  Images path    : {cls.KVASIR_IMG_DIR}")

        print(f"  Masks path     : {cls.KVASIR_MASK_DIR}")

        print(f"  Split          : {int(cls.TRAIN_RATIO*100)}% Train / {int(cls.VAL_RATIO*100)}% Val / {int(cls.TEST_RATIO*100)}% Test")

        print(f"  Image Size     : {cls.IMG_SIZE}×{cls.IMG_SIZE}")

        print(f"  Batch Size     : {cls.BATCH_SIZE}")

        print(f"  Epochs         : {cls.NUM_EPOCHS}")

        print(f"  Checkpoints    : {cls.CHECKPOINT_DIR}")

        print("=" * 60)



    @classmethod

    def verify_paths(cls):

        """Call this to confirm paths exist before training."""

        print("\nVerifying dataset paths...")

        img_ok  = os.path.isdir(cls.KVASIR_IMG_DIR)

        mask_ok = os.path.isdir(cls.KVASIR_MASK_DIR)

       

        n_imgs  = len(os.listdir(cls.KVASIR_IMG_DIR))  if img_ok  else 0

        n_masks = len(os.listdir(cls.KVASIR_MASK_DIR)) if mask_ok else 0

       

        print(f"  {'✓' if img_ok  else '✗'}  images: {cls.KVASIR_IMG_DIR}  ({n_imgs} files)")

        print(f"  {'✓' if mask_ok else '✗'}  masks:  {cls.KVASIR_MASK_DIR}  ({n_masks} files)")

       

        if not img_ok or not mask_ok:

            print("\n  ✗ PATH NOT FOUND. Check your 'data' folder structure.")

            print(f"    Expected: {cls.DATASET_ROOT}")

        else:

            print(f"\n  ✓ All paths verified. Ready to train.")

       

        return img_ok and mask_ok and n_imgs > 0



# Initialize config object to be imported elsewhere

cfg = Config()
