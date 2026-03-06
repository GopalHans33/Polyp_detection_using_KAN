import os

import cv2

import numpy as np

import torch

from glob import glob

from torch.utils.data import Dataset, DataLoader

import albumentations as A

from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split



# ══════════════════════════════════════════════════════════════════════════════

#  1. Augmentation Pipelines (Advanced)

# ══════════════════════════════════════════════════════════════════════════════



def get_train_transform(img_size=256):

    """

    Heavy augmentation for training.

    Helps the model generalize to different lighting, orientations, and noise.

    """

    return A.Compose([

        A.Resize(img_size, img_size),

       

        # Geometric (Flips/Rotations)

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),

       

        # Elastic & Distortion (Simulates tissue deformation)

        A.OneOf([

            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),

            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5),

            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),

        ], p=0.3),



        # Color & Lighting (Simulates different endoscope lights)

        A.OneOf([

            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),

            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),

        ], p=0.5),



        # Noise & Blur

        A.OneOf([

            A.GaussNoise(p=0.5),

            A.MotionBlur(blur_limit=5, p=0.5),

            A.MedianBlur(blur_limit=5, p=0.5),

        ], p=0.2),



        # Normalization (Required for ResNet/ImageNet weights)

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ToTensorV2(),

    ])



def get_val_transform(img_size=256):

    """Validation transform: Resize and Normalize only."""

    return A.Compose([

        A.Resize(img_size, img_size),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ToTensorV2(),

    ])



# ══════════════════════════════════════════════════════════════════════════════

#  2. Boundary Helper

# ══════════════════════════════════════════════════════════════════════════════



def compute_boundary(mask_np, kernel_size=5):

    """

    Generates the boundary map for the loss function.

    boundary = dilate(mask) - erode(mask)

    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

   

    # Ensure mask is uint8 for OpenCV

    mask_uint8 = (mask_np * 255).astype(np.uint8) if mask_np.max() <= 1.0 else mask_np.astype(np.uint8)

   

    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

    eroded  = cv2.erode(mask_uint8,  kernel, iterations=1)

   

    boundary = dilated.astype(np.float32) - eroded.astype(np.float32)

    return np.clip(boundary / 255.0, 0.0, 1.0)



# ══════════════════════════════════════════════════════════════════════════════

#  3. Dataset Class

# ══════════════════════════════════════════════════════════════════════════════



class KvasirDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None, img_size=256, boundary_kernel=5):

        self.image_paths = image_paths

        self.mask_paths  = mask_paths

        self.transform   = transform

        self.img_size    = img_size

        self.bk_size     = boundary_kernel



    def __len__(self):

        return len(self.image_paths)



    def __getitem__(self, idx):

        # 1. Load Image

        img_path = self.image_paths[idx]

        image = cv2.imread(img_path)

        if image is None:

            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        # 2. Load Mask

        mask_path = self.mask_paths[idx]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:

            raise FileNotFoundError(f"Mask not found: {mask_path}")



        # Binarize mask (0 or 1)

        mask = (mask > 127).astype(np.float32)



        # 3. Apply Albumentations

        if self.transform:

            # Albumentations expects mask in uint8 for geometric ops, or float for others

            # We pass it as is, but ensure consistency

            augmented = self.transform(image=image, mask=mask)

            image = augmented['image'] # Tensor [3, H, W]

            mask  = augmented['mask']  # Tensor [H, W]



            # 4. Compute Boundary from the AUGMENTED mask

            # Convert back to numpy for CV2 operations

            mask_np = mask.numpy()

            bnd_np  = compute_boundary(mask_np, self.bk_size)

           

            # Convert back to tensor

            bnd_mask = torch.from_numpy(bnd_np).unsqueeze(0) # [1, H, W]

           

            # Ensure mask is [1, H, W]

            mask = mask.unsqueeze(0)



        else:

            # Manual resize if no transform (fallback)

            image = cv2.resize(image, (self.img_size, self.img_size))

            mask  = cv2.resize(mask,  (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

           

            bnd_np = compute_boundary(mask, self.bk_size)

           

            # Normalize Image manually

            mean = np.array([0.485, 0.456, 0.406])

            std  = np.array([0.229, 0.224, 0.225])

            image = (image.astype(np.float32) / 255.0 - mean) / std

            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

           

            mask     = torch.from_numpy(mask).unsqueeze(0).float()

            bnd_mask = torch.from_numpy(bnd_np).unsqueeze(0).float()



        return image, mask, bnd_mask



# ══════════════════════════════════════════════════════════════════════════════

#  4. Data Loader Builder

# ══════════════════════════════════════════════════════════════════════════════



def build_dataloaders(cfg):

    """

    Scans the Kvasir folder, performs a stratified split, and returns dataloaders.

    Returns: [train_loader, val_loader, test_loader]

    """

    print("\n[Dataset] -------------------------------------")

    print(f"[Dataset] Scanning: {cfg.KVASIR_IMG_DIR}")

   

    # 1. Scan Directories

    # Support multiple extensions

    exts = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']

    all_img_paths = []

    for ext in exts:

        all_img_paths.extend(glob(os.path.join(cfg.KVASIR_IMG_DIR, ext)))

    all_img_paths = sorted(list(set(all_img_paths))) # Remove duplicates and sort

   

    if not all_img_paths:

        raise ValueError(f"No images found in {cfg.KVASIR_IMG_DIR}. Check config paths!")



    # Find matching masks

    valid_imgs = []

    valid_masks = []

   

    for img_path in all_img_paths:

        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Try finding mask with common extensions

        found_mask = None

        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:

            possible_path = os.path.join(cfg.KVASIR_MASK_DIR, basename + ext)

            if os.path.exists(possible_path):

                found_mask = possible_path

                break

       

        if found_mask:

            valid_imgs.append(img_path)

            valid_masks.append(found_mask)



    print(f"[Dataset] Found {len(valid_imgs)} valid image-mask pairs.")



    # 2. Stratified Split (based on polyp size)

    # This ensures we don't get a train set with only empty masks

    strata = []

    for mp in valid_masks:

        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        if m is None:

            strata.append(0)

            continue

        coverage = (m > 127).sum() / m.size

        # Simple discretization for stratification

        if coverage == 0: cat = 0

        elif coverage < 0.1: cat = 1

        elif coverage < 0.2: cat = 2

        else: cat = 3

        strata.append(cat)



    # Split: Train (80%), Temp (20%)

    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(

        valid_imgs, valid_masks, train_size=cfg.TRAIN_RATIO,

        stratify=strata, random_state=cfg.SEED

    )

   

    # Split Temp: Val (10%), Test (10%)

    # We need to re-calculate strata for the temp set to stratify again

    strata_temp = []

    for mp in temp_masks:

        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        coverage = (m > 127).sum() / m.size if m is not None else 0

        if coverage == 0: cat = 0

        elif coverage < 0.1: cat = 1

        elif coverage < 0.2: cat = 2

        else: cat = 3

        strata_temp.append(cat)



    val_imgs, test_imgs, val_masks, test_masks = train_test_split(

        temp_imgs, temp_masks, test_size=0.5,

        stratify=strata_temp, random_state=cfg.SEED

    )



    print(f"[Dataset] Split: {len(train_imgs)} Train, {len(val_imgs)} Val, {len(test_imgs)} Test")



    # 3. Create Datasets

    train_ds = KvasirDataset(train_imgs, train_masks, transform=get_train_transform(cfg.IMG_SIZE), img_size=cfg.IMG_SIZE)

    val_ds   = KvasirDataset(val_imgs,   val_masks,   transform=get_val_transform(cfg.IMG_SIZE),   img_size=cfg.IMG_SIZE)

    test_ds  = KvasirDataset(test_imgs,  test_masks,  transform=get_val_transform(cfg.IMG_SIZE),   img_size=cfg.IMG_SIZE)



    # 4. Create Loaders

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,

                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True)

    val_loader   = DataLoader(val_ds,   batch_size=cfg.VAL_BATCH_SIZE, shuffle=False,

                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

    test_loader  = DataLoader(test_ds,  batch_size=cfg.VAL_BATCH_SIZE, shuffle=False,

                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)



    print("[Dataset] Ready.")

   

    # Return a LIST so it is easy to index

    return [train_loader, val_loader, test_loader]
