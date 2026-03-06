"""
dataset.py  —  KA-ResUNet++
=============================
Kvasir-SEG dataset, split, and dataloaders.

FIXES IN THIS VERSION:
  [FIX 1] build_dataloaders now returns exactly 3 values (train, val, test).
           Previously returned 5 (train, val, test, None, None) which caused:
           ValueError: too many values to unpack (expected 3) in main.py.

  [FIX 2] ImageCompression: auto-detects albumentations API version.
           Old albumentations: quality_range=(10, 40)
           New albumentations: quality_lower=10, quality_upper=40
           Code now works with either version installed in Colab.

  [FIX 3] CoarseDropout: auto-detects albumentations API version.
           Old albumentations: max_holes, max_height, max_width, min_holes, fill_value
           New albumentations: num_holes_range, hole_height_range, hole_width_range
           Code now works with either version installed in Colab.

  [FIX 4] ElasticTransform: alpha_affine argument removed (deprecated >= 1.4).
  [FIX 5] OpticalDistortion: shift_limit argument removed (deprecated >= 1.4).
"""

import os
import cv2
import numpy as np
import warnings
from glob import glob
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════════════════════
#  Albumentations version-safe helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_image_compression():
    """
    Returns ImageCompression transform that works with any albumentations version.
    Old: quality_range=(10, 40)
    New: quality_lower=10, quality_upper=40
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            t = A.ImageCompression(quality_lower=10, quality_upper=40, p=0.3)
        return t
    except Exception:
        return A.ImageCompression(quality_range=(10, 40), p=0.3)


def _make_coarse_dropout():
    """
    Returns CoarseDropout transform that works with any albumentations version.
    Old: max_holes, max_height, max_width, min_holes, fill_value
    New: num_holes_range, hole_height_range, hole_width_range
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            t = A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, 32),
                hole_width_range=(1, 32),
                p=0.3
            )
        return t
    except Exception:
        return A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, fill_value=0, p=0.3
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Augmentation Pipelines
# ══════════════════════════════════════════════════════════════════════════════

def get_train_transform(img_size: int = 256) -> A.Compose:
    """Full augmentation pipeline. Compatible with any albumentations version."""
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1,
            rotate_limit=15, p=0.5
        ),

        # Elastic deformations — deprecated args removed
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=12, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.2, p=0.5),
        ], p=0.4),

        # Blur / noise
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
            A.Defocus(radius=(1, 3), p=0.5),
        ], p=0.3),
        A.GaussNoise(p=0.3),

        # FIX 2: version-safe ImageCompression
        _make_image_compression(),

        # Lighting
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),

        # FIX 3: version-safe CoarseDropout
        _make_coarse_dropout(),

        # Normalize with ImageNet stats
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 256) -> A.Compose:
    """Validation/test transform — resize and normalize only."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  Boundary Map
# ══════════════════════════════════════════════════════════════════════════════

def compute_boundary(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """dilate(mask) - erode(mask). Returns float32 boundary ring in [0,1]."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded  = cv2.erode(mask,  kernel, iterations=1)
    return np.clip(dilated.astype(np.float32) - eroded.astype(np.float32), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  KvasirDataset
# ══════════════════════════════════════════════════════════════════════════════

class KvasirDataset(Dataset):
    """
    Kvasir-SEG dataset. Images and masks share the same filename.
    Returns (image [3,H,W], seg_mask [1,H,W], boundary_mask [1,H,W]).
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths:  List[str],
        transform:   Optional[A.Compose] = None,
        img_size:    int = 256,
        boundary_kernel_size: int = 5,
    ):
        assert len(image_paths) == len(mask_paths), \
            f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform
        self.img_size    = img_size
        self.bk_size     = boundary_kernel_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Load
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Cannot read: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read: {self.mask_paths[idx]}")

        # Binarize before augmentation
        seg_mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(
                image=image,
                mask=(seg_mask * 255).astype(np.uint8)
            )
            image   = augmented["image"]
            seg_aug = augmented["mask"].float() / 255.0
            seg_np  = seg_aug.numpy()
            bnd_np  = compute_boundary((seg_np > 0.5).astype(np.float32), self.bk_size)
            seg_mask = seg_aug.unsqueeze(0)
            bnd_mask = torch.from_numpy(bnd_np).unsqueeze(0)
        else:
            image    = cv2.resize(image, (self.img_size, self.img_size))
            seg_mask = cv2.resize(seg_mask, (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_NEAREST)
            bnd_np   = compute_boundary((seg_mask > 0.5).astype(np.float32), self.bk_size)
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            image    = (image.astype(np.float32) / 255.0 - mean) / std
            image    = torch.from_numpy(image.transpose(2, 0, 1)).float()
            seg_mask = torch.from_numpy(seg_mask).unsqueeze(0).float()
            bnd_mask = torch.from_numpy(bnd_np).unsqueeze(0).float()

        return image, seg_mask, bnd_mask


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _scan_kvasir(img_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
    """Match images to masks by base filename."""
    exts = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")

    img_paths = []
    for ext in exts:
        img_paths.extend(glob(os.path.join(img_dir, f"*.{ext}")))
    img_paths = sorted(set(img_paths))

    mask_lookup = {}
    for ext in exts:
        for p in glob(os.path.join(mask_dir, f"*.{ext}")):
            key = os.path.splitext(os.path.basename(p))[0]
            mask_lookup[key] = p

    matched_imgs, matched_masks = [], []
    missing = 0
    for ip in img_paths:
        key = os.path.splitext(os.path.basename(ip))[0]
        if key in mask_lookup:
            matched_imgs.append(ip)
            matched_masks.append(mask_lookup[key])
        else:
            missing += 1

    if missing > 0:
        print(f"  [Warning] {missing} images had no matching mask — skipped.")

    return matched_imgs, matched_masks


def _get_coverage_strata(mask_paths: List[str]) -> List[str]:
    """Assign polyp coverage category for stratified splitting."""
    strata = []
    for mp in mask_paths:
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            strata.append("small")
            continue
        c = (mask > 127).sum() / mask.size
        if   c == 0:    strata.append("empty")
        elif c <= 0.05: strata.append("small")
        elif c <= 0.15: strata.append("medium")
        elif c <= 0.30: strata.append("large")
        else:           strata.append("huge")
    return strata


# ══════════════════════════════════════════════════════════════════════════════
#  build_dataloaders — returns exactly 3 loaders
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(cfg):
    """
    Build train / val / test DataLoaders from Kvasir-SEG.

    Split (stratified by polyp size):
        800 train  |  100 val  |  100 test

    FIX: Returns exactly 3 values now — train_loader, val_loader, test_loader.
    Previously returned 5 values (with two trailing Nones), which caused:
        ValueError: too many values to unpack (expected 3)
    in any main.py that does: train_loader, val_loader, test_loader = build_dataloaders(cfg)
    """
    print(f"\n[Dataset] {'─'*45}")
    print(f"[Dataset] Scanning: {cfg.KVASIR_IMG_DIR}")

    all_imgs, all_masks = _scan_kvasir(cfg.KVASIR_IMG_DIR, cfg.KVASIR_MASK_DIR)

    if len(all_imgs) == 0:
        raise RuntimeError(
            f"\nNo images found in: {cfg.KVASIR_IMG_DIR}\n"
            f"Check that cfg.KVASIR_IMG_DIR is set correctly in config.py."
        )

    print(f"[Dataset] Found {len(all_imgs)} valid image-mask pairs.")

    strata = _get_coverage_strata(all_masks)

    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        all_imgs, all_masks,
        test_size=(cfg.VAL_RATIO + cfg.TEST_RATIO),
        random_state=cfg.SEED, stratify=strata,
    )
    strata_temp = _get_coverage_strata(temp_masks)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks,
        test_size=0.5,
        random_state=cfg.SEED, stratify=strata_temp,
    )

    print(f"[Dataset] Split: {len(train_imgs)} Train, {len(val_imgs)} Val, {len(test_imgs)} Test")

    train_dataset = KvasirDataset(
        train_imgs, train_masks,
        get_train_transform(cfg.IMG_SIZE),
        cfg.IMG_SIZE, cfg.BOUNDARY_KERNEL_SIZE,
    )
    val_dataset = KvasirDataset(
        val_imgs, val_masks,
        get_val_transform(cfg.IMG_SIZE),
        cfg.IMG_SIZE, cfg.BOUNDARY_KERNEL_SIZE,
    )
    test_dataset = KvasirDataset(
        test_imgs, test_masks,
        get_val_transform(cfg.IMG_SIZE),
        cfg.IMG_SIZE, cfg.BOUNDARY_KERNEL_SIZE,
    )

    print("[Dataset] Ready.")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=True,  num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
    )

    # FIX: return exactly 3 values — no trailing Nones
    return train_loader, val_loader, test_loader


def verify_batch(loader: DataLoader, name: str = "loader"):
    """Sanity check — print shapes and value ranges of first batch."""
    for imgs, seg_masks, bnd_masks in loader:
        print(f"  [{name}]")
        print(f"    image : {tuple(imgs.shape)}      — should be (B, 3, 256, 256)")
        print(f"    seg   : {tuple(seg_masks.shape)} — should be (B, 1, 256, 256)")
        print(f"    bnd   : {tuple(bnd_masks.shape)} — should be (B, 1, 256, 256)")
        print(f"    img range  : [{imgs.min():.2f}, {imgs.max():.2f}]")
        print(f"    seg values : {seg_masks.unique().tolist()}")
        pct = (seg_masks > 0.5).float().mean().item() * 100
        print(f"    polyp coverage: {pct:.1f}%")
        break
