# idrid_fundus.py
"""
IDRiD multi-task fundus model:
- DR grading (0–4)
- DME risk (0–2)

This file is structured so that:

- When imported (e.g. by fusion_app.py):
    -> Only model classes, constants, and helper functions are defined.
    -> NO data loading, preprocessing, training, or evaluation runs.

- When run as a script: `python idrid_fundus.py`
    -> Full pipeline executes:
       - Load raw CSVs
       - Preprocess images (cached to disk & NPZ)
       - Train multi-task model
       - Calibrate with temperature scaling
       - Evaluate on val + test
       - Save calibrated checkpoint: checkpoints/best_idrid_simple_calibrated.pt
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import timm

# ------------------------------------------------------------------------
# 1. Global config (used both in training and by external code)
# ------------------------------------------------------------------------

# Raw IDRiD data (only needed when running training script)
# NOTE: These paths point to the original dataset location.
# For training from scratch, download IDRiD dataset and place in ../Datasets/Disease Grading/
# For inference only (fusion_app.py), these paths are not needed - only the trained model is used.
TRAIN_DIR = Path("../Datasets/Original Images/Training Set")
TEST_DIR  = Path("../Datasets/Original Images/Testing Set")
TRAIN_CSV = Path("../Datasets/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
TEST_CSV  = Path("../Datasets/Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")

# Preprocessed images / CSVs / NPZ
PREP_ROOT       = Path("preprocessed_idrid")
PREP_TRAIN_DIR  = PREP_ROOT / "train"
PREP_TEST_DIR   = PREP_ROOT / "test"
PREP_TRAIN_CSV  = PREP_ROOT / "train_preprocessed.csv"
PREP_TEST_CSV   = PREP_ROOT / "test_preprocessed.csv"

TRAIN_NPZ = PREP_ROOT / "train_arrays.npz"
TEST_NPZ  = PREP_ROOT / "test_arrays.npz"

# Outputs
RUN_DIR   = Path("runs/idrid_simple")
CKPT_PATH = Path("checkpoints/best_idrid_simple.pt")
CAL_CKPT_PATH = Path("checkpoints/best_idrid_simple_calibrated.pt")

# Model / training
MODEL_NAME = "mobilenetv3_large_100"
IMG_SIZE   = 320

NUM_DR_CLASSES  = 5  # DR grade: 0..4
NUM_DME_CLASSES = 3  # DME risk: 0..2

LOSS_WEIGHT_DME = 1.0

BATCH_SIZE   = 12
EPOCHS       = 20
VAL_SPLIT    = 0.2
LR_WARMUP    = 4e-4
LR_FINETUNE  = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 7

IMBALANCE_STRATEGY = "sampler"  # "sampler", "class_weights", "none"
WARMUP_EPOCHS      = 2

SEED = 24

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Albumentations offline augmentation flags (run only when training and set True)
DO_DR1_AUG               = False  # targeted DR=1 augmentation (one-time)
N_AUG_PER_DR1            = 4
USE_ALBUMENTATIONS_OFFLINE = False  # general offline aug (one-time)

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# ------------------------------------------------------------------------
# 2. Helper functions (safe to use when imported)
# ------------------------------------------------------------------------

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        key = c.lower().strip()
        if key in lowered:
            return lowered[key]
    raise ValueError(f"CSV missing expected columns. Tried {candidates}. Found: {list(df.columns)}")


def fundus_bbox_square(img: np.ndarray, pad_ratio: float = 0.01):
    """
    Robust crop that tries not to cut off the fundus:
    1) mask non-dark pixels
    2) bounding rect of mask
    3) pad to square with black borders
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thr = max(5, int(np.percentile(gray, 5)))
        mask = (gray > thr).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        H, W = img.shape[:2]
        padx = int(w * pad_ratio)
        pady = int(h * pad_ratio)
        x1 = max(0, x - padx)
        y1 = max(0, y - pady)
        x2 = min(W, x + w + padx)
        y2 = min(H, y + h + pady)
        crop = img[y1:y2, x1:x2]

        hh, ww = crop.shape[:2]
        if hh == ww:
            return crop
        side = max(hh, ww)
        top = (side - hh) // 2
        bottom = side - hh - top
        left = (side - ww) // 2
        right = side - ww - left
        crop_sq = cv2.copyMakeBorder(
            crop, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return crop_sq
    except Exception:
        return img


def retina_enhance(rgb: np.ndarray):
    """Simple Ben Graham style shade correction + mild unsharp mask."""
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=rgb.shape[1] * 0.05)
    out = cv2.addWeighted(rgb, 4.0, blur, -4.0, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------
# 3. Model definitions (safe to import)
# ------------------------------------------------------------------------

class MultiTaskEffNet(nn.Module):
    """
    Multi-task network:
      - Shared backbone (timm mobilenetv3-large-100)
      - Head for DR grade (num_dr classes)
      - Head for DME risk (num_dme classes)
    """
    def __init__(self, backbone_name, num_dr, num_dme, drop_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            drop_rate=drop_rate,
        )
        feat_dim = self.backbone.num_features
        self.head_dr  = nn.Linear(feat_dim, num_dr)
        self.head_dme = nn.Linear(feat_dim, num_dme)

    def forward(self, x):
        feats = self.backbone(x)
        logits_dr  = self.head_dr(feats)
        logits_dme = self.head_dme(feats)
        return logits_dr, logits_dme


class TemperatureScaler(nn.Module):
    """
    Simple temperature scaling module:
    logits_scaled = logits / T, where T = exp(log_temp) > 0
    """
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))  # T starts at 1.0

    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T


# ------------------------------------------------------------------------
# 4. Main training / calibration pipeline
#    (only runs when __name__ == "__main__")
# ------------------------------------------------------------------------

def main():
    # 4.1 Setup
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is written for GPU training only.")
    device = torch.device("cuda:0")
    print(f"[Info] Using device: {device}")

    # 4.2 Load original CSVs
    df_train_raw = pd.read_csv(TRAIN_CSV)
    df_test_raw  = pd.read_csv(TEST_CSV)

    df_train = df_train_raw[["Image name", "Retinopathy grade", "Risk of macular edema"]].copy()
    df_train.columns = ["image", "label_dr", "label_dme"]

    df_test = df_test_raw[["Image name", "Retinopathy grade", "Risk of macular edema "]].copy()
    df_test.columns = ["image", "label_dr", "label_dme"]

    def ensure_ext(x):
        x = str(x)
        if any(x.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]):
            return x
        return x + ".jpg"

    df_train["image"]    = df_train["image"].apply(ensure_ext)
    df_test["image"]     = df_test["image"].apply(ensure_ext)

    df_train["label_dr"]  = df_train["label_dr"].astype(int)
    df_train["label_dme"] = df_train["label_dme"].astype(int)
    df_test["label_dr"]   = df_test["label_dr"].astype(int)
    df_test["label_dme"]  = df_test["label_dme"].astype(int)

    print(f"[Info] Original -> Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    # 4.3 Preprocess & save images (only once, cached by CSVs)
    PREP_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PREP_TEST_DIR.mkdir(parents=True, exist_ok=True)
    PREP_ROOT.mkdir(parents=True, exist_ok=True)

    if PREP_TRAIN_CSV.exists() and PREP_TEST_CSV.exists():
        print(f"[Info] Preprocessed CSVs already exist: {PREP_TRAIN_CSV}, {PREP_TEST_CSV}")
        df_train_prep = pd.read_csv(PREP_TRAIN_CSV)
        df_test_prep  = pd.read_csv(PREP_TEST_CSV)
    else:
        print("[Info] Preprocessing images (this is the slow part, done once)...")

        # TRAIN
        paths_prep      = []
        labels_dr_prep  = []
        labels_dme_prep = []
        for i, row in df_train.iterrows():
            if (i + 1) % 50 == 0:
                print(f"  [Train] {i+1}/{len(df_train)}")
            raw_path = Path(TRAIN_DIR) / row["image"]
            if not raw_path.exists():
                print(f"  [WARN] Missing train image: {raw_path}")
                continue
            bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  [WARN] Cannot read train image: {raw_path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = fundus_bbox_square(rgb, pad_ratio=0.01)
            rgb = retina_enhance(rgb)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            out_name = f"train_{i:04d}.jpg"
            out_path = PREP_TRAIN_DIR / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            paths_prep.append(str(out_path.resolve()))
            labels_dr_prep.append(int(row["label_dr"]))
            labels_dme_prep.append(int(row["label_dme"]))

        df_train_prep = pd.DataFrame({
            "path": paths_prep,
            "label_dr": labels_dr_prep,
            "label_dme": labels_dme_prep,
        })
        df_train_prep.to_csv(PREP_TRAIN_CSV, index=False)
        print(f"[Info] Saved preprocessed train CSV: {PREP_TRAIN_CSV}")

        # TEST
        paths_prep      = []
        labels_dr_prep  = []
        labels_dme_prep = []
        for i, row in df_test.iterrows():
            if (i + 1) % 50 == 0:
                print(f"  [Test] {i+1}/{len(df_test)}")
            raw_path = Path(TEST_DIR) / row["image"]
            if not raw_path.exists():
                print(f"  [WARN] Missing test image: {raw_path}")
                continue
            bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  [WARN] Cannot read test image: {raw_path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = fundus_bbox_square(rgb, pad_ratio=0.01)
            rgb = retina_enhance(rgb)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            out_name = f"test_{i:04d}.jpg"
            out_path = PREP_TEST_DIR / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            paths_prep.append(str(out_path.resolve()))
            labels_dr_prep.append(int(row["label_dr"]))
            labels_dme_prep.append(int(row["label_dme"]))

        df_test_prep = pd.DataFrame({
            "path": paths_prep,
            "label_dr": labels_dr_prep,
            "label_dme": labels_dme_prep,
        })
        df_test_prep.to_csv(PREP_TEST_CSV, index=False)
        print(f"[Info] Saved preprocessed test CSV: {PREP_TEST_CSV}")

    # 4.4 Optional targeted DR=1 augmentation (one-time)
    if DO_DR1_AUG:
        import albumentations as A
        print(f"[Info] Running targeted augmentation for DR grade-1 ({N_AUG_PER_DR1} extra per original).")
        dr1_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.05,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT101,
                p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

        df_dr1 = df_train_prep[df_train_prep["label_dr"] == 1].reset_index(drop=True)
        print(f"[Info] Found {len(df_dr1)} DR grade-1 images for augmentation.")
        extra_rows = []
        for i, row in df_dr1.iterrows():
            img_path = Path(row["path"])
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] Cannot read DR1 image: {img_path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for k in range(N_AUG_PER_DR1):
                augmented = dr1_aug(image=rgb)["image"]
                out_name = f"{img_path.stem}_dr1aug{k}.jpg"
                out_path = PREP_TRAIN_DIR / out_name
                cv2.imwrite(str(out_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                extra_rows.append({
                    "path": str(out_path.resolve()),
                    "label_dr": int(row["label_dr"]),
                    "label_dme": int(row["label_dme"]),
                })
            if (i + 1) % 10 == 0:
                print(f"  [DR1-Aug] {i+1}/{len(df_dr1)} originals done")

        if extra_rows:
            df_extra = pd.DataFrame(extra_rows)
            df_train_prep = pd.concat([df_train_prep, df_extra], ignore_index=True)
            df_train_prep.to_csv(PREP_TRAIN_CSV, index=False)
            print(f"[Info] Added {len(df_extra)} augmented DR=1 samples. New train size: {len(df_train_prep)}")

    # 4.5 Optional generic offline Albumentations (one-time)
    if USE_ALBUMENTATIONS_OFFLINE:
        import albumentations as A
        print("[Info] Running offline Albumentations to create extra augmented images...")
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.05,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT101,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

        extra_paths      = []
        extra_labels_dr  = []
        extra_labels_dme = []
        for i, row in df_train_prep.iterrows():
            if (i + 1) % 50 == 0:
                print(f"  [Aug] {i+1}/{len(df_train_prep)}")
            img_path = Path(row["path"])
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            augmented = aug(image=rgb)["image"]
            out_name = img_path.stem + "_aug.jpg"
            out_path = PREP_TRAIN_DIR / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            extra_paths.append(str(out_path.resolve()))
            extra_labels_dr.append(int(row["label_dr"]))
            extra_labels_dme.append(int(row["label_dme"]))

        if extra_paths:
            df_extra = pd.DataFrame({
                "path": extra_paths,
                "label_dr": extra_labels_dr,
                "label_dme": extra_labels_dme,
            })
            df_train_prep = pd.concat([df_train_prep, df_extra], ignore_index=True)
            df_train_prep.to_csv(PREP_TRAIN_CSV, index=False)
            print(f"[Info] Albumentations added {len(extra_paths)} samples. New train size: {len(df_train_prep)}")

    # 4.6 Load NPZ or build arrays
    if TRAIN_NPZ.exists() and TEST_NPZ.exists():
        print(f"[Info] NPZ cache found: {TRAIN_NPZ}, {TEST_NPZ}")
        train_data = np.load(TRAIN_NPZ)
        test_data  = np.load(TEST_NPZ)
        X_train     = train_data["X"]
        y_train_dr  = train_data["y_dr"]
        y_train_dme = train_data["y_dme"]
        X_test      = test_data["X"]
        y_test_dr   = test_data["y_dr"]
        y_test_dme  = test_data["y_dme"]
    else:
        print("[Info] NPZ cache not found. Loading JPGs and building arrays (one-time)...")
        # Train
        X_list      = []
        y_dr_list   = []
        y_dme_list  = []
        for i, row in df_train_prep.iterrows():
            if (i + 1) % 100 == 0:
                print(f"  [Load-Train] {i+1}/{len(df_train_prep)}")
            path = row["path"]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] Cannot read preprocessed train image: {path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            X_list.append(rgb)
            y_dr_list.append(int(row["label_dr"]))
            y_dme_list.append(int(row["label_dme"]))
        X_train     = np.stack(X_list, axis=0)
        y_train_dr  = np.array(y_dr_list, dtype=np.int64)
        y_train_dme = np.array(y_dme_list, dtype=np.int64)

        # Test
        X_list      = []
        y_dr_list   = []
        y_dme_list  = []
        for i, row in df_test_prep.iterrows():
            if (i + 1) % 100 == 0:
                print(f"  [Load-Test] {i+1}/{len(df_test_prep)}")
            path = row["path"]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] Cannot read preprocessed test image: {path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            X_list.append(rgb)
            y_dr_list.append(int(row["label_dr"]))
            y_dme_list.append(int(row["label_dme"]))
        X_test      = np.stack(X_list, axis=0)
        y_test_dr   = np.array(y_dr_list, dtype=np.int64)
        y_test_dme  = np.array(y_dme_list, dtype=np.int64)

        np.savez(TRAIN_NPZ, X=X_train, y_dr=y_train_dr, y_dme=y_train_dme)
        np.savez(TEST_NPZ,  X=X_test,  y_dr=y_test_dr,  y_dme=y_test_dme)
        print(f"[Info] Saved NPZ cache: {TRAIN_NPZ}, {TEST_NPZ}")

    print(f"[Info] X_train shape: {X_train.shape}, y_train_dr shape: {y_train_dr.shape}, y_train_dme shape: {y_train_dme.shape}")
    print(f"[Info] X_test shape:  {X_test.shape},  y_test_dr shape:  {y_test_dr.shape},  y_test_dme shape:  {y_test_dme.shape}")

    # 4.7 Normalize and convert to tensors
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0

    mean = IMAGENET_MEAN.reshape(1, 1, 1, 3)
    std  = IMAGENET_STD.reshape(1, 1, 1, 3)
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test  = np.transpose(X_test,  (0, 3, 1, 2))

    X_train_t    = torch.from_numpy(X_train)
    y_train_dr_t = torch.from_numpy(y_train_dr)
    y_train_dme_t= torch.from_numpy(y_train_dme)
    X_test_t     = torch.from_numpy(X_test)
    y_test_dr_t  = torch.from_numpy(y_test_dr)
    y_test_dme_t = torch.from_numpy(y_test_dme)

    print(f"[Info] Tensor shapes -> X_train: {X_train_t.shape}, X_test: {X_test_t.shape}")

    # 4.8 Train/val split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_SPLIT,
        random_state=SEED,
    )
    train_idx, val_idx = next(splitter.split(np.arange(len(y_train_dr)), y_train_dr))

    X_tr_t        = X_train_t[train_idx]
    y_tr_dr_t     = y_train_dr_t[train_idx]
    y_tr_dme_t    = y_train_dme_t[train_idx]
    X_va_t        = X_train_t[val_idx]
    y_va_dr_t     = y_train_dr_t[val_idx]
    y_va_dme_t    = y_train_dme_t[val_idx]

    print(f"[Info] Split -> Train: {len(y_tr_dr_t)}, Val: {len(y_va_dr_t)}, Test: {len(y_test_dr_t)}")

    train_dataset = TensorDataset(X_tr_t, y_tr_dr_t, y_tr_dme_t)
    val_dataset   = TensorDataset(X_va_t, y_va_dr_t, y_va_dme_t)
    test_dataset  = TensorDataset(X_test_t, y_test_dr_t, y_test_dme_t)

    # Class imbalance handling
    sampler = None
    class_weights_dr = None

    if IMBALANCE_STRATEGY == "sampler":
        counts = np.bincount(y_tr_dr_t.numpy(), minlength=NUM_DR_CLASSES)
        w_per_sample = 1.0 / (counts[y_tr_dr_t.numpy()] + 1e-6)
        sampler = WeightedRandomSampler(
            weights=w_per_sample,
            num_samples=len(w_per_sample),
            replacement=True,
        )
        print("[Info] Using WeightedRandomSampler (DR labels).")
    elif IMBALANCE_STRATEGY == "class_weights":
        counts = np.bincount(y_tr_dr_t.numpy(), minlength=NUM_DR_CLASSES).astype(float)
        weights = counts.sum() / (counts + 1e-6)
        weights = weights / weights.mean()
        class_weights_dr = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[Info] Using DR class weights: {class_weights_dr.cpu().numpy()}")
    else:
        print("[Info] No imbalance strategy.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 4.9 Model, loss, optimizer
    model = MultiTaskEffNet(MODEL_NAME, NUM_DR_CLASSES, NUM_DME_CLASSES, drop_rate=0.4).to(device)
    print(f"[Info] Multi-task model created. First param device: {next(model.parameters()).device}")

    criterion_dr  = nn.CrossEntropyLoss(weight=class_weights_dr)
    criterion_dme = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_WARMUP,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = None
    scaler = torch.amp.GradScaler('cuda')

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    waited = 0

    # 4.10 Training loop
    for epoch in range(1, EPOCHS + 1):
        # Warmup -> Unfreeze backbone
        if epoch == WARMUP_EPOCHS + 1:
            print(f"[Info] Warmup finished. Unfreezing backbone for finetuning (LR={LR_FINETUNE}).")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LR_FINETUNE,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS - WARMUP_EPOCHS,
                eta_min=LR_FINETUNE * 0.1,
            )

        # TRAIN
        model.train()
        total_loss = 0.0
        total_samples = 0
        for step, (imgs, labels_dr, labels_dme) in enumerate(train_loader):
            imgs       = imgs.to(device, non_blocking=True)
            labels_dr  = labels_dr.to(device, non_blocking=True)
            labels_dme = labels_dme.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=True):
                logits_dr, logits_dme = model(imgs)
                loss_dr  = criterion_dr(logits_dr, labels_dr)
                loss_dme = criterion_dme(logits_dme, labels_dme)
                loss = loss_dr + LOSS_WEIGHT_DME * loss_dme

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            if (step + 1) % 10 == 0:
                print(f"  [Epoch {epoch:02d}] step {step+1}/{len(train_loader)} loss={loss.item():.4f}")

        train_loss = total_loss / max(total_samples, 1)

        # VALIDATION
        model.eval()
        all_dr_labels  = []
        all_dr_preds   = []
        all_dme_labels = []
        all_dme_preds  = []
        with torch.no_grad():
            for imgs, labels_dr, labels_dme in val_loader:
                imgs       = imgs.to(device, non_blocking=True)
                labels_dr  = labels_dr.to(device, non_blocking=True)
                labels_dme = labels_dme.to(device, non_blocking=True)

                logits_dr, logits_dme = model(imgs)
                preds_dr  = logits_dr.argmax(dim=1)
                preds_dme = logits_dme.argmax(dim=1)

                all_dr_labels.append(labels_dr.cpu())
                all_dr_preds.append(preds_dr.cpu())
                all_dme_labels.append(labels_dme.cpu())
                all_dme_preds.append(preds_dme.cpu())

        all_dr_labels  = torch.cat(all_dr_labels).numpy()
        all_dr_preds   = torch.cat(all_dr_preds).numpy()
        all_dme_labels = torch.cat(all_dme_labels).numpy()
        all_dme_preds  = torch.cat(all_dme_preds).numpy()

        val_acc_dr = accuracy_score(all_dr_labels, all_dr_preds)
        val_f1_dr  = f1_score(all_dr_labels, all_dr_preds, average="macro")
        val_acc_dme = accuracy_score(all_dme_labels, all_dme_preds)
        val_f1_dme  = f1_score(all_dme_labels, all_dme_preds, average="macro")

        print(
            f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  "
            f"val_f1_DR={val_f1_dr:.4f}  val_acc_DR={val_acc_dr:.4f}  "
            f"val_f1_DME={val_f1_dme:.4f}  val_acc_DME={val_acc_dme:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        avg_f1 = 0.5 * (val_f1_dr + val_f1_dme)
        if avg_f1 > best_f1 + 1e-5:
            best_f1 = avg_f1
            waited = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1_dr":  float(val_f1_dr),
                    "val_f1_dme": float(val_f1_dme),
                },
                CKPT_PATH,
            )
            print("  ↳ Saved checkpoint")
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"[EarlyStop] No improvement for {PATIENCE} epochs.")
                break

    # 4.11 Load best checkpoint
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Info] Loaded best checkpoint from epoch {ckpt['epoch']} | "
              f"val_f1_dr={ckpt.get('val_f1_dr', float('nan')):.4f}  "
              f"val_f1_dme={ckpt.get('val_f1_dme', float('nan')):.4f}")

    # 4.12 Collect val logits for temperature scaling
    model.eval()
    val_logits_dr_list  = []
    val_logits_dme_list = []
    val_labels_dr_list  = []
    val_labels_dme_list = []

    with torch.no_grad():
        for imgs, labels_dr, labels_dme in val_loader:
            imgs       = imgs.to(device, non_blocking=True)
            labels_dr  = labels_dr.to(device, non_blocking=True)
            labels_dme = labels_dme.to(device, non_blocking=True)

            logits_dr, logits_dme = model(imgs)
            val_logits_dr_list.append(logits_dr)
            val_logits_dme_list.append(logits_dme)
            val_labels_dr_list.append(labels_dr)
            val_labels_dme_list.append(labels_dme)

    val_logits_dr  = torch.cat(val_logits_dr_list,  dim=0)
    val_logits_dme = torch.cat(val_logits_dme_list, dim=0)
    val_labels_dr  = torch.cat(val_labels_dr_list,  dim=0)
    val_labels_dme = torch.cat(val_labels_dme_list, dim=0)

    print("[Info] Collected validation logits for calibration:",
          val_logits_dr.shape, val_logits_dme.shape)

    temp_dr  = TemperatureScaler().to(device)
    temp_dme = TemperatureScaler().to(device)
    ce_loss = nn.CrossEntropyLoss()

    def fit_temperature(temp_module, logits, labels, max_iter=500, lr=0.01):
        optimizer_t = torch.optim.Adam(temp_module.parameters(), lr=lr)
        temp_module.train()
        for _ in range(max_iter):
            optimizer_t.zero_grad()
            scaled_logits = temp_module(logits)
            loss = ce_loss(scaled_logits, labels)
            loss.backward()
            optimizer_t.step()
        temp_module.eval()
        return temp_module

    print("[Info] Fitting temperature for DR head...")
    temp_dr = fit_temperature(temp_dr, val_logits_dr, val_labels_dr)
    print("[Info] Fitting temperature for DME head...")
    temp_dme = fit_temperature(temp_dme, val_logits_dme, val_labels_dme)

    print("Learned temperature DR:",  torch.exp(temp_dr.log_temp).item())
    print("Learned temperature DME:", torch.exp(temp_dme.log_temp).item())

    # 4.13 Validation report with calibration
    def evaluate(loader, name):
        model.eval()
        all_dr_labels  = []
        all_dr_preds   = []
        all_dme_labels = []
        all_dme_preds  = []
        with torch.no_grad():
            for imgs, labels_dr, labels_dme in loader:
                imgs       = imgs.to(device, non_blocking=True)
                labels_dr  = labels_dr.to(device, non_blocking=True)
                labels_dme = labels_dme.to(device, non_blocking=True)

                logits_dr, logits_dme = model(imgs)
                logits_dr  = temp_dr(logits_dr)
                logits_dme = temp_dme(logits_dme)

                probs_dr  = torch.softmax(logits_dr, dim=1)
                probs_dme = torch.softmax(logits_dme, dim=1)

                preds_dr  = probs_dr.argmax(dim=1)
                preds_dme = probs_dme.argmax(dim=1)

                all_dr_labels.append(labels_dr.cpu())
                all_dr_preds.append(preds_dr.cpu())
                all_dme_labels.append(labels_dme.cpu())
                all_dme_preds.append(preds_dme.cpu())

        all_dr_labels  = torch.cat(all_dr_labels).numpy()
        all_dr_preds   = torch.cat(all_dr_preds).numpy()
        all_dme_labels = torch.cat(all_dme_labels).numpy()
        all_dme_preds  = torch.cat(all_dme_preds).numpy()

        report_dr  = classification_report(all_dr_labels,  all_dr_preds,  digits=4)
        cm_dr      = confusion_matrix(all_dr_labels,       all_dr_preds)
        acc_dr     = accuracy_score(all_dr_labels,         all_dr_preds)
        f1_dr      = f1_score(all_dr_labels,               all_dr_preds,  average="macro")

        report_dme = classification_report(all_dme_labels, all_dme_preds, digits=4)
        cm_dme     = confusion_matrix(all_dme_labels,      all_dme_preds)
        acc_dme    = accuracy_score(all_dme_labels,        all_dme_preds)
        f1_dme     = f1_score(all_dme_labels,              all_dme_preds, average="macro")

        print(f"\n===== {name} results (DR) =====")
        print(report_dr)
        print(f"\n===== {name} results (DME) =====")
        print(report_dme)

        return (acc_dr, f1_dr, cm_dr), (acc_dme, f1_dme, cm_dme)

    (val_acc_dr, val_f1_dr, val_cm_dr), (val_acc_dme, val_f1_dme, val_cm_dme) = evaluate(val_loader, "Validation")
    (test_acc_dr, test_f1_dr, test_cm_dr), (test_acc_dme, test_f1_dme, test_cm_dme) = evaluate(test_loader, "Test")

    # 4.14 Save calibrated checkpoint
    CAL_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "temp_dr_state_dict": temp_dr.state_dict(),
            "temp_dme_state_dict": temp_dme.state_dict(),
        },
        CAL_CKPT_PATH,
    )
    print(f"[Info] Saved calibrated model + temperatures to {CAL_CKPT_PATH}")

    # 4.15 Save metrics
    metrics_out = {
        "val": {
            "dr": {
                "acc": float(val_acc_dr),
                "f1_macro": float(val_f1_dr),
                "confusion_matrix": val_cm_dr.tolist(),
            },
            "dme": {
                "acc": float(val_acc_dme),
                "f1_macro": float(val_f1_dme),
                "confusion_matrix": val_cm_dme.tolist(),
            },
        },
        "test": {
            "dr": {
                "acc": float(test_acc_dr),
                "f1_macro": float(test_f1_dr),
                "confusion_matrix": test_cm_dr.tolist(),
            },
            "dme": {
                "acc": float(test_acc_dme),
                "f1_macro": float(test_f1_dme),
                "confusion_matrix": test_cm_dme.tolist(),
            },
        },
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "metrics_simple.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"[Info] Metrics saved to {RUN_DIR / 'metrics_simple.json'}")
    print("[Done] Training + calibration + evaluation complete.")


# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------

if __name__ == "__main__":
    main()
