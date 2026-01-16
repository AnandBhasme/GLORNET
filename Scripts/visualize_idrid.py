# %% [markdown]
# 0. Imports & basic setup

# %%
import os, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support

import timm
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# %%
def list_available_models(pattern=None):
    """
    List all available timm models, optionally filtered by pattern.
    
    Usage:
        list_available_models()  # List all models
        list_available_models("efficientnet")  # List only EfficientNet models
        list_available_models("convnext")  # List only ConvNeXt models
    """
    all_models = timm.list_models(pretrained=True)
    if pattern:
        filtered = [m for m in all_models if pattern.lower() in m.lower()]
        print(f"[Info] Found {len(filtered)} models matching '{pattern}':")
        for model in sorted(filtered):
            print(f"  {model}")
    else:
        print(f"[Info] Total available models: {len(all_models)}")
        print("[Info] Use list_available_models('pattern') to filter, e.g.:")
        print("  list_available_models('efficientnet')")
        print("  list_available_models('convnext')")
        print("  list_available_models('resnet')")
    return all_models if not pattern else filtered

# %%
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# %% [markdown]
# 1. Config (easy to tweak)

# %%
# NOTE: These paths point to the original dataset location.
# For visualization, download IDRiD dataset and place in ../Datasets/Disease Grading/
TRAIN_DIR = r"../Datasets/Original Images/Training Set"
TEST_DIR  = r"../Datasets/Original Images/Testing Set"
TRAIN_CSV = r"../Datasets/Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
TEST_CSV  = r"../Datasets/Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"


# ---- Paths (preprocessed) ----
PREP_ROOT       = Path("preprocessed_idrid")
PREP_TRAIN_DIR  = PREP_ROOT / "train"
PREP_TEST_DIR   = PREP_ROOT / "test"
PREP_TRAIN_CSV  = PREP_ROOT / "train_preprocessed.csv"
PREP_TEST_CSV   = PREP_ROOT / "test_preprocessed.csv"

# ---- Outputs ----
RUN_DIR   = Path("runs/idrid_simple")
CKPT_PATH = Path("checkpoints/best_idrid_simple.pt")

# Model / training
MODEL_NAME = "convnext_tiny"   
IMG_SIZE   = 320                      # smaller than 512 → faster
#NUM_CLASSES = 5

NUM_DR_CLASSES = 5          # Retinopathy grade: 0..4
NUM_DME_CLASSES = 3         # Risk of macular edema: 0..2

LOSS_WEIGHT_DME = 1.0       # you can tune this (e.g. 0.5) if one task dominates

BATCH_SIZE = 12
EPOCHS = 20
VAL_SPLIT = 0.2
LR_WARMUP = 4e-4               # new
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7

IMBALANCE_STRATEGY = "sampler"         # "sampler", "class_weights", "none"
WARMUP_EPOCHS = 2 

SEED = 24


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# %%
TRAIN_NPZ = PREP_ROOT / "train_arrays.npz"
TEST_NPZ  = PREP_ROOT / "test_arrays.npz"

# %% [markdown]
# 1. Basic helpers (minimal functions)

# %%
def set_seed(seed=42):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed(SEED)

# %%
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

# %%
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

# %%
def retina_enhance(rgb: np.ndarray):
    """Simple Ben Graham style shade correction + mild unsharp mask."""
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=rgb.shape[1] * 0.05)
    out = cv2.addWeighted(rgb, 4.0, blur, -4.0, 128)
    return np.clip(out, 0, 255).astype(np.uint8)

# %% [markdown]
# 2. Init seed & device

# %%
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script is written for GPU training only.")
device = torch.device("cuda:0")
print(f"[Info] Using device: {device}")

# %% [markdown]
# 3. Read original CSVs

# %%
df_train_raw = pd.read_csv(TRAIN_CSV)
df_test_raw  = pd.read_csv(TEST_CSV)

# %%




# train_img_col  = find_column(df_train_raw, img_col_candidates)
# train_label_col = find_column(df_train_raw, label_col_candidates)
# test_img_col   = find_column(df_test_raw, img_col_candidates)
# test_label_col  = find_column(df_test_raw, label_col_candidates)



# %%
df_train = df_train_raw[["Image name", "Retinopathy grade", "Risk of macular edema"]].copy()
df_train.columns = ["image", "label_dr", "label_dme"]

df_test = df_test_raw[["Image name", "Retinopathy grade", "Risk of macular edema "]].copy()
df_test.columns = ["image", "label_dr", "label_dme"]

# %%
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

# %%
print(f"[Info] Original -> Train rows: {len(df_train)}, Test rows: {len(df_test)}")

# %% [markdown]
# 4. Preprocess & save images (only once)
# - crop, enhance, resize
# - save to PREP_TRAIN_DIR / PREP_TEST_DIR
# - write new CSVs with full paths

# %%
df_train.iterrows

# %%
PREP_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
PREP_TEST_DIR.mkdir(parents=True, exist_ok=True)
PREP_ROOT.mkdir(parents=True, exist_ok=True)

if PREP_TRAIN_CSV.exists() and PREP_TEST_CSV.exists():
    print(f"[Info] Preprocessed CSVs already exist: {PREP_TRAIN_CSV}, {PREP_TEST_CSV}")
    df_train_prep = pd.read_csv(PREP_TRAIN_CSV)
    df_test_prep  = pd.read_csv(PREP_TEST_CSV)
else:
    print("[Info] Preprocessing images (this is the slow part, done once)...")

    # ---- preprocess TRAIN ----
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

        # crop + enhance + resize
        rgb = fundus_bbox_square(rgb, pad_ratio=0.01)
        rgb = retina_enhance(rgb)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # save
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

# %%
# ---- preprocess TEST ----
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

# %%
# ============================================================
# 4b. Targeted offline augmentation for DR grade-1 ONLY
#     (run ONCE; then comment out this block or set flag False)
# ============================================================
DO_DR1_AUG = False          # set False after you’ve run once
N_AUG_PER_DR1 = 4          # 3–5 is reasonable

if DO_DR1_AUG:
    import albumentations as A

    print(f"[Info] Running targeted augmentation for DR grade-1 "
          f"({N_AUG_PER_DR1} extra samples per original).")

    # mild, realistic augmentations
    dr1_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.02,
                           scale_limit=0.05,
                           rotate_limit=10,
                           border_mode=cv2.BORDER_REFLECT101,
                           p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2)
    ])

    # filter DR grade-1 rows
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
                "label_dr": int(row["label_dr"]),      # always 1
                "label_dme": int(row["label_dme"])     # keep same DME label
            })

        if (i + 1) % 10 == 0:
            print(f"  [DR1-Aug] {i+1}/{len(df_dr1)} originals done")

    if extra_rows:
        df_extra = pd.DataFrame(extra_rows)
        df_train_prep = pd.concat([df_train_prep, df_extra], ignore_index=True)
        df_train_prep.to_csv(PREP_TRAIN_CSV, index=False)
        print(f"[Info] Added {len(df_extra)} augmented DR=1 samples. "
              f"New train size: {len(df_train_prep)}")
    else:
        print("[Info] No extra DR1 samples created (check warnings above).")


# %% [markdown]
# 5. OPTIONAL: Albumentations offline augmentation

# %%
USE_ALBUMENTATIONS_OFFLINE = False  # <-- set True if you want to run this block ONCE

# %%
import albumentations as A

print("[Info] Running offline Albumentations to create extra augmented images...")
aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10, border_mode=cv2.BORDER_REFLECT101, p=0.5),
    A.GaussianBlur(blur_limit=(3,5), p=0.2)
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

# %%
class TemperatureScaler(nn.Module):
    """
    Simple temperature scaling module:
    logits_scaled = logits / T, where T = exp(log_temp) > 0
    """
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))  # T starts at 1.0

    def forward(self, logits):
        # logits: (N, C)
        T = torch.exp(self.log_temp)
        return logits / T


# %% [markdown]
# 6. Load preprocessed images into memory & normalize

# %%
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

    # ---- build train arrays ----
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

    # ---- build test arrays ----
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

    # save NPZ so future runs are INSTANT
    np.savez(TRAIN_NPZ, X=X_train, y_dr=y_train_dr, y_dme=y_train_dme)
    np.savez(TEST_NPZ,  X=X_test,  y_dr=y_test_dr,  y_dme=y_test_dme)
    print(f"[Info] Saved NPZ cache: {TRAIN_NPZ}, {TEST_NPZ}")

print(f"[Info] X_train shape: {X_train.shape}, y_train_dr shape: {y_train_dr.shape}, y_train_dme shape: {y_train_dme.shape}")
print(f"[Info] X_test shape:  {X_test.shape},  y_test_dr shape:  {y_test_dr.shape},  y_test_dme shape:  {y_test_dme.shape}")


# %%
# def load_images_from_df(df, img_size):
#     X_list      = []
#     y_dr_list   = []
#     y_dme_list  = []
#     for i, row in df.iterrows():
#         if (i + 1) % 100 == 0:
#             print(f"  [Load] {i+1}/{len(df)}")
#         path = row["path"]
#         bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
#         if bgr is None:
#             print(f"[WARN] Cannot read preprocessed image: {path}")
#             continue
#         rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#         rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
#         X_list.append(rgb)
#         y_dr_list.append(int(row["label_dr"]))
#         y_dme_list.append(int(row["label_dme"]))

#     X    = np.stack(X_list, axis=0)
#     y_dr = np.array(y_dr_list, dtype=np.int64)
#     y_dme = np.array(y_dme_list, dtype=np.int64)
#     return X, y_dr, y_dme

# print("[Info] Loading preprocessed train images into RAM...")
# X_train, y_train_dr, y_train_dme = load_images_from_df(df_train_prep, IMG_SIZE)

# print("[Info] Loading preprocessed test images into RAM...")
# X_test, y_test_dr, y_test_dme   = load_images_from_df(df_test_prep, IMG_SIZE)

# %%
# print(f"[Info] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"[Info] X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")

# %%
X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32)  / 255.0

# %%
mean = IMAGENET_MEAN.reshape(1, 1, 1, 3)
std  = IMAGENET_STD.reshape(1, 1, 1, 3)
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# %%
# N,H,W,C -> N,C,H,W
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test  = np.transpose(X_test,  (0, 3, 1, 2))

# %%
X_train_t    = torch.from_numpy(X_train)
y_train_dr_t = torch.from_numpy(y_train_dr)
y_train_dme_t = torch.from_numpy(y_train_dme)

X_test_t     = torch.from_numpy(X_test)
y_test_dr_t  = torch.from_numpy(y_test_dr)
y_test_dme_t = torch.from_numpy(y_test_dme)

# %%
print(f"[Info] Tensor shapes -> X_train: {X_train_t.shape}, X_test: {X_test_t.shape}")

# %% [markdown]
# 7. Train/val split

# %%
splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=VAL_SPLIT,
    random_state=SEED
)
train_idx, val_idx = next(splitter.split(np.arange(len(y_train_dr)), y_train_dr))


# %%
X_tr_t        = X_train_t[train_idx]
y_tr_dr_t     = y_train_dr_t[train_idx]
y_tr_dme_t    = y_train_dme_t[train_idx]

X_va_t        = X_train_t[val_idx]
y_va_dr_t     = y_train_dr_t[val_idx]
y_va_dme_t    = y_train_dme_t[val_idx]

print(f"[Info] Split -> Train: {len(y_tr_dr_t)}, Val: {len(y_va_dr_t)}, Test: {len(y_test_dr_t)}")

# %% [markdown]
# 8. Dataloaders (TensorDataset, no custom Dataset class)

# %%
train_dataset = TensorDataset(X_tr_t, y_tr_dr_t, y_tr_dme_t)
val_dataset   = TensorDataset(X_va_t, y_va_dr_t, y_va_dme_t)
test_dataset  = TensorDataset(X_test_t, y_test_dr_t, y_test_dme_t)


# %%
sampler = None
class_weights_dr = None

# %%
IMBALANCE_STRATEGY = "sampler"

# %%
if IMBALANCE_STRATEGY == "sampler":
    counts = np.bincount(y_tr_dr_t.numpy(), minlength=NUM_DR_CLASSES)
    w_per_sample = 1.0 / (counts[y_tr_dr_t.numpy()] + 1e-6)
    sampler = WeightedRandomSampler(
        weights=w_per_sample,
        num_samples=len(w_per_sample),
        replacement=True
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


# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# %% [markdown]
# 9. Model, loss, optimizer

# %%
# model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, drop_rate=0.4)
# model = model.to(device)
# print(f"[Info] Model '{MODEL_NAME}' created. First param device: {next(model.parameters()).device}")


# # --- Warmup setup: freeze all but classifier head ---
# WARMUP_EPOCHS = WARMUP_EPOCHS  # uses the value you set above

# # 1) Freeze everything
# for p in model.parameters():
#     p.requires_grad = False

# # 2) Unfreeze only classifier / head parameters
# for name, p in model.named_parameters():
#     if any(k in name.lower() for k in ["classifier", "head", "fc"]):
#         p.requires_grad = True

# print("[Info] Warmup: training only the classifier head for first "
#       f"{WARMUP_EPOCHS} epochs.")


# %%
class MultiTaskEffNet(nn.Module):
    def __init__(self, backbone_name, num_dr, num_dme, drop_rate=0.4):
        super().__init__()
        # backbone without classifier, global_pool='avg' for pooled features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg',
            drop_rate=drop_rate
        )
        feat_dim = self.backbone.num_features
        self.head_dr  = nn.Linear(feat_dim, num_dr)
        self.head_dme = nn.Linear(feat_dim, num_dme)

    def forward(self, x):
        feats = self.backbone(x)
        logits_dr  = self.head_dr(feats)
        logits_dme = self.head_dme(feats)
        return logits_dr, logits_dme

model = MultiTaskEffNet(MODEL_NAME, NUM_DR_CLASSES, NUM_DME_CLASSES, drop_rate=0.4).to(device)
print(f"[Info] Multi-task model created. First param device: {next(model.parameters()).device}")


# %%
def count_parameters(model):
    """Count total, trainable, and non-trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params

total_params, trainable_params, non_trainable_params = count_parameters(model)
print(f"[Info] Model parameters:")
print(f"  Total params: {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Non-trainable params: {non_trainable_params:,}")

# %%
criterion_dr  = nn.CrossEntropyLoss(weight=class_weights_dr)
criterion_dme = nn.CrossEntropyLoss()   # could add weights if DME is imbalanced

# Warmup optimizer: head-only
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_WARMUP,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS - WARMUP_EPOCHS,
    eta_min=LR_FINETUNE * 0.1,
)



# %%
scaler = torch.amp.GradScaler('cuda')

# %% [markdown]
# 10. Training loop (inline, no extra functions)

# %%
best_f1 = -1.0
waited = 0

RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

# %%
scheduler = None   # start with no scheduler

# Lists to track metrics across epochs for plotting
train_losses = []
train_accs_dr = []
train_accs_dme = []
val_losses = []
val_accs_dr = []
val_accs_dme = []

for epoch in range(1, EPOCHS + 1):

    # -------------------------------
    # WARMUP → UNFREEZE
    # -------------------------------
    if epoch == WARMUP_EPOCHS + 1:
        print(f"[Info] Warmup finished. Unfreezing backbone for finetuning (LR={LR_FINETUNE}).")

        for p in model.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR_FINETUNE,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS - WARMUP_EPOCHS,
            eta_min=LR_FINETUNE * 0.1
        )

    # -------------------------------
    # TRAIN
    # -------------------------------
    model.train()
    total_loss = 0.0
    total_samples = 0
    train_dr_labels = []
    train_dr_preds = []
    train_dme_labels = []
    train_dme_preds = []

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

        # Track predictions for accuracy
        preds_dr = logits_dr.argmax(dim=1)
        preds_dme = logits_dme.argmax(dim=1)
        train_dr_labels.append(labels_dr.cpu())
        train_dr_preds.append(preds_dr.cpu())
        train_dme_labels.append(labels_dme.cpu())
        train_dme_preds.append(preds_dme.cpu())

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if (step + 1) % 10 == 0:
            print(f"  [Epoch {epoch:02d}] step {step+1}/{len(train_loader)} loss={loss.item():.4f}")

    train_loss = total_loss / max(total_samples, 1)

    # Compute train accuracy
    train_dr_labels_all = torch.cat(train_dr_labels).numpy()
    train_dr_preds_all = torch.cat(train_dr_preds).numpy()
    train_dme_labels_all = torch.cat(train_dme_labels).numpy()
    train_dme_preds_all = torch.cat(train_dme_preds).numpy()
    
    train_acc_dr = accuracy_score(train_dr_labels_all, train_dr_preds_all)
    train_acc_dme = accuracy_score(train_dme_labels_all, train_dme_preds_all)

    # -------------------------------
    # VALIDATION
    # -------------------------------
    model.eval()
    all_dr_labels  = []
    all_dr_preds   = []
    all_dme_labels = []
    all_dme_preds  = []
    val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        for imgs, labels_dr, labels_dme in val_loader:
            imgs       = imgs.to(device, non_blocking=True)
            labels_dr  = labels_dr.to(device, non_blocking=True)
            labels_dme = labels_dme.to(device, non_blocking=True)

            logits_dr, logits_dme = model(imgs)

            # Compute loss
            loss_dr = criterion_dr(logits_dr, labels_dr)
            loss_dme = criterion_dme(logits_dme, labels_dme)
            loss = loss_dr + LOSS_WEIGHT_DME * loss_dme
            val_loss += loss.item() * imgs.size(0)
            val_samples += imgs.size(0)

            preds_dr  = logits_dr.argmax(dim=1)
            preds_dme = logits_dme.argmax(dim=1)

            all_dr_labels.append(labels_dr.cpu())
            all_dr_preds.append(preds_dr.cpu())
            all_dme_labels.append(labels_dme.cpu())
            all_dme_preds.append(preds_dme.cpu())

    val_loss = val_loss / max(val_samples, 1)
    all_dr_labels  = torch.cat(all_dr_labels).numpy()
    all_dr_preds   = torch.cat(all_dr_preds).numpy()
    all_dme_labels = torch.cat(all_dme_labels).numpy()
    all_dme_preds  = torch.cat(all_dme_preds).numpy()

    val_acc_dr = accuracy_score(all_dr_labels, all_dr_preds)
    val_f1_dr  = f1_score(all_dr_labels, all_dr_preds, average="macro")

    val_acc_dme = accuracy_score(all_dme_labels, all_dme_preds)
    val_f1_dme  = f1_score(all_dme_labels, all_dme_preds, average="macro")

        # Store metrics for plotting
    train_losses.append(train_loss)
    train_accs_dr.append(train_acc_dr)
    train_accs_dme.append(train_acc_dme)
    val_losses.append(val_loss)
    val_accs_dr.append(val_acc_dr)
    val_accs_dme.append(val_acc_dme)

    print(
        f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  train_acc_DR={train_acc_dr:.4f}  train_acc_DME={train_acc_dme:.4f}  "
        f"val_loss={val_loss:.4f}  val_f1_DR={val_f1_dr:.4f}  val_acc_DR={val_acc_dr:.4f}  "
        f"val_f1_DME={val_f1_dme:.4f}  val_acc_DME={val_acc_dme:.4f}"
    )

    # -------------------------------
    # SCHEDULER STEP
    # -------------------------------
    if scheduler is not None:
        scheduler.step()

    # -------------------------------
    # EARLY STOP / CHECKPOINT
    # -------------------------------
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


# %% [markdown]
# 11. Load best and full evaluation (val + test)

# %%
epochs = range(1, len(train_losses) + 1)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

# Plot 1: Loss vs Epochs
axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Model Loss vs Epochs', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: DR Accuracy vs Epochs
axes[0, 1].plot(epochs, train_accs_dr, 'b-', label='Train Accuracy (DR)', linewidth=2)
axes[0, 1].plot(epochs, val_accs_dr, 'r-', label='Validation Accuracy (DR)', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('DR Accuracy vs Epochs', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1])

# Plot 3: DME Accuracy vs Epochs
axes[1, 0].plot(epochs, train_accs_dme, 'b-', label='Train Accuracy (DME)', linewidth=2)
axes[1, 0].plot(epochs, val_accs_dme, 'r-', label='Validation Accuracy (DME)', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Accuracy', fontsize=12)
axes[1, 0].set_title('DME Accuracy vs Epochs', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1])

# Plot 4: Combined Accuracy (Average of DR and DME)
train_accs_combined = [(dr + dme) / 2 for dr, dme in zip(train_accs_dr, train_accs_dme)]
val_accs_combined = [(dr + dme) / 2 for dr, dme in zip(val_accs_dr, val_accs_dme)]
axes[1, 1].plot(epochs, train_accs_combined, 'b-', label='Train Accuracy (Avg)', linewidth=2)
axes[1, 1].plot(epochs, val_accs_combined, 'r-', label='Validation Accuracy (Avg)', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Accuracy', fontsize=12)
axes[1, 1].set_title('Combined Accuracy vs Epochs', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(RUN_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
print(f"[Info] Training curves saved to {RUN_DIR / 'training_curves.png'}")
plt.show()

# %%
if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Info] Loaded best checkpoint from epoch {ckpt['epoch']} | "
          f"val_f1_dr={ckpt.get('val_f1_dr', float('nan')):.4f}  "
          f"val_f1_dme={ckpt.get('val_f1_dme', float('nan')):.4f}")



# %%
# 11b. Collect validation logits and labels for calibration
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

# Concatenate into big tensors
val_logits_dr  = torch.cat(val_logits_dr_list,  dim=0)
val_logits_dme = torch.cat(val_logits_dme_list, dim=0)
val_labels_dr  = torch.cat(val_labels_dr_list,  dim=0)
val_labels_dme = torch.cat(val_labels_dme_list, dim=0)

print("[Info] Collected validation logits for calibration:",
      val_logits_dr.shape, val_logits_dme.shape)


# %%
# 11c. Fit temperature scalers for DR and DME
temp_dr  = TemperatureScaler().to(device)
temp_dme = TemperatureScaler().to(device)

ce_loss = nn.CrossEntropyLoss()

def fit_temperature(temp_module, logits, labels, max_iter=500, lr=0.01):
    optimizer = torch.optim.Adam(temp_module.parameters(), lr=lr)
    temp_module.train()
    for i in range(max_iter):
        optimizer.zero_grad()
        scaled_logits = temp_module(logits)
        loss = ce_loss(scaled_logits, labels)
        loss.backward()
        optimizer.step()
    temp_module.eval()
    return temp_module

print("[Info] Fitting temperature for DR head...")
temp_dr = fit_temperature(temp_dr, val_logits_dr, val_labels_dr)

print("[Info] Fitting temperature for DME head...")
temp_dme = fit_temperature(temp_dme, val_logits_dme, val_labels_dme)

print("Learned temperature DR:",  torch.exp(temp_dr.log_temp).item())
print("Learned temperature DME:", torch.exp(temp_dme.log_temp).item())


# %%
# Validation report (DR + DME)
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

        # --- apply temperature scaling ---
        logits_dr  = temp_dr(logits_dr)
        logits_dme = temp_dme(logits_dme)

        # Now softmax if you want probabilities, or just argmax for class labels
        probs_dr = torch.softmax(logits_dr, dim=1)
        probs_dme = torch.softmax(logits_dme, dim=1)

        preds_dr  = probs_dr.argmax(dim=1)
        preds_dme = probs_dme.argmax(dim=1)


all_dr_labels  = torch.cat(all_dr_labels).numpy()
all_dr_preds   = torch.cat(all_dr_preds).numpy()
all_dme_labels = torch.cat(all_dme_labels).numpy()
all_dme_preds  = torch.cat(all_dme_preds).numpy()

val_report_dr  = classification_report(all_dr_labels,  all_dr_preds,  digits=4)
val_cm_dr      = confusion_matrix(all_dr_labels,       all_dr_preds)
val_acc_dr     = accuracy_score(all_dr_labels,         all_dr_preds)
val_f1_dr      = f1_score(all_dr_labels,               all_dr_preds,  average="macro")

val_report_dme = classification_report(all_dme_labels, all_dme_preds, digits=4)
val_cm_dme     = confusion_matrix(all_dme_labels,      all_dme_preds)
val_acc_dme    = accuracy_score(all_dme_labels,        all_dme_preds)
val_f1_dme     = f1_score(all_dme_labels,              all_dme_preds, average="macro")

print("\n===== Validation results (DR) =====")
print(val_report_dr)

print("\n===== Validation results (DME) =====")
print(val_report_dme)


# %%
# Test report (DR + DME)
all_dr_labels  = []
all_dr_preds   = []
all_dme_labels = []
all_dme_preds  = []

with torch.no_grad():
    for imgs, labels_dr, labels_dme in test_loader:
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


test_report_dr  = classification_report(all_dr_labels,  all_dr_preds,  digits=4)
test_cm_dr      = confusion_matrix(all_dr_labels,       all_dr_preds)
test_acc_dr     = accuracy_score(all_dr_labels,         all_dr_preds)
test_f1_dr      = f1_score(all_dr_labels,               all_dr_preds,  average="macro")

test_report_dme = classification_report(all_dme_labels, all_dme_preds, digits=4)
test_cm_dme     = confusion_matrix(all_dme_labels,      all_dme_preds)
test_acc_dme    = accuracy_score(all_dme_labels,        all_dme_preds)
test_f1_dme     = f1_score(all_dme_labels,              all_dme_preds, average="macro")


print("===== Test results (DR) =====")
print(test_report_dr)

print("===== Test results (DME) =====")
print(test_report_dme)


# %%
# ==========================
# Validation report (DR + DME)
# ==========================
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

        # Forward pass
        logits_dr, logits_dme = model(imgs)

        # Apply temperature scaling
        logits_dr  = temp_dr(logits_dr)
        logits_dme = temp_dme(logits_dme)

        # Get probabilities (if needed) and predictions
        probs_dr  = torch.softmax(logits_dr, dim=1)
        probs_dme = torch.softmax(logits_dme, dim=1)

        preds_dr  = probs_dr.argmax(dim=1)
        preds_dme = probs_dme.argmax(dim=1)

        # Collect labels and predictions on CPU
        all_dr_labels.append(labels_dr.cpu())
        all_dr_preds.append(preds_dr.cpu())
        all_dme_labels.append(labels_dme.cpu())
        all_dme_preds.append(preds_dme.cpu())

# Concatenate all batches and convert to numpy arrays
val_dr_labels  = torch.cat(all_dr_labels).numpy()
val_dr_preds   = torch.cat(all_dr_preds).numpy()
val_dme_labels = torch.cat(all_dme_labels).numpy()
val_dme_preds  = torch.cat(all_dme_preds).numpy()

# Compute metrics
val_report_dr  = classification_report(val_dr_labels,  val_dr_preds,  digits=4)
val_cm_dr      = confusion_matrix(val_dr_labels,       val_dr_preds)
val_acc_dr     = accuracy_score(val_dr_labels,         val_dr_preds)
val_f1_dr      = f1_score(val_dr_labels,               val_dr_preds,  average="macro")

val_report_dme = classification_report(val_dme_labels, val_dme_preds, digits=4)
val_cm_dme     = confusion_matrix(val_dme_labels,      val_dme_preds)
val_acc_dme    = accuracy_score(val_dme_labels,        val_dme_preds)
val_f1_dme     = f1_score(val_dme_labels,              val_dme_preds, average="macro")

print("\n===== Validation results (DR) =====")
print(val_report_dr)

print("\n===== Validation results (DME) =====")
print(val_report_dme)


# ======================
# Test report (DR + DME)
# ======================
model.eval()
all_dr_labels  = []
all_dr_preds   = []
all_dme_labels = []
all_dme_preds  = []

with torch.no_grad():
    for imgs, labels_dr, labels_dme in test_loader:
        imgs       = imgs.to(device, non_blocking=True)
        labels_dr  = labels_dr.to(device, non_blocking=True)
        labels_dme = labels_dme.to(device, non_blocking=True)

        # Forward pass
        logits_dr, logits_dme = model(imgs)

        # Apply temperature scaling
        logits_dr  = temp_dr(logits_dr)
        logits_dme = temp_dme(logits_dme)

        probs_dr  = torch.softmax(logits_dr, dim=1)
        probs_dme = torch.softmax(logits_dme, dim=1)

        preds_dr  = probs_dr.argmax(dim=1)
        preds_dme = probs_dme.argmax(dim=1)

        # Collect labels and predictions on CPU
        all_dr_labels.append(labels_dr.cpu())
        all_dr_preds.append(preds_dr.cpu())
        all_dme_labels.append(labels_dme.cpu())
        all_dme_preds.append(preds_dme.cpu())

# Concatenate and convert to numpy
test_dr_labels  = torch.cat(all_dr_labels).numpy()
test_dr_preds   = torch.cat(all_dr_preds).numpy()
test_dme_labels = torch.cat(all_dme_labels).numpy()
test_dme_preds  = torch.cat(all_dme_preds).numpy()

# Compute metrics
test_report_dr  = classification_report(test_dr_labels,  test_dr_preds,  digits=4)
test_cm_dr      = confusion_matrix(test_dr_labels,       test_dr_preds)
test_acc_dr     = accuracy_score(test_dr_labels,         test_dr_preds)
test_f1_dr      = f1_score(test_dr_labels,               test_dr_preds,  average="macro")

test_report_dme = classification_report(test_dme_labels, test_dme_preds, digits=4)
test_cm_dme     = confusion_matrix(test_dme_labels,      test_dme_preds)
test_acc_dme    = accuracy_score(test_dme_labels,        test_dme_preds)
test_f1_dme     = f1_score(test_dme_labels,              test_dme_preds, average="macro")

print("\n===== Test results (DR) =====")
print(test_report_dr)

print("\n===== Test results (DME) =====")
print(test_report_dme)

# %% [markdown]
# Comprehensive Visualizations

# %%
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", figsize=(12, 5)):
    """Plot an enhanced heatmap of the confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages for better interpretation
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent = np.nan_to_num(cm_percent, nan=0.0)  # Handle division by zero
    
    # Create DataFrames
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {c}" for c in class_names],
        columns=[f"Pred: {c}" for c in class_names]
    )
    
    cm_percent_df = pd.DataFrame(
        cm_percent,
        index=[f"True: {c}" for c in class_names],
        columns=[f"Pred: {c}" for c in class_names]
    )
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Raw counts
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                ax=axes[0], cbar_kws={'label': 'Count'}, linewidths=0.5)
    axes[0].set_title(f'{title} (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Plot 2: Percentages
    sns.heatmap(cm_percent_df, annot=True, fmt='.1f', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5)
    axes[1].set_title(f'{title} (Percentages)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return cm

# %%
# DR Class names
dr_class_names = [f"DR Grade {i}" for i in range(NUM_DR_CLASSES)]
dme_class_names = [f"DME Risk {i}" for i in range(NUM_DME_CLASSES)]

print("\n===== Confusion Matrix Visualizations =====")

# Validation Confusion Matrices
print("\n[Validation Set]")
plot_confusion_matrix(val_dr_labels, val_dr_preds, dr_class_names, 
                     "Validation Set - DR Confusion Matrix (Calibrated)")
plot_confusion_matrix(val_dme_labels, val_dme_preds, dme_class_names,
                     "Validation Set - DME Confusion Matrix (Calibrated)")

# Test Confusion Matrices  
print("\n[Test Set]")
plot_confusion_matrix(test_dr_labels, test_dr_preds, dr_class_names,
                     "Test Set - DR Confusion Matrix (Calibrated)")
plot_confusion_matrix(test_dme_labels, test_dme_preds, dme_class_names,
                     "Test Set - DME Confusion Matrix (Calibrated)")

# %%
# Class-wise Performance Visualization
def plot_class_wise_metrics(y_true, y_pred, class_names, title_prefix="", figsize=(15, 5)):
    """Plot precision, recall, and F1-score for each class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Precision
    sns.barplot(data=metrics_df, x='Class', y='Precision', ax=axes[0], palette='Blues')
    axes[0].set_title(f'{title_prefix}Precision by Class', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Recall
    sns.barplot(data=metrics_df, x='Class', y='Recall', ax=axes[1], palette='Greens')
    axes[1].set_title(f'{title_prefix}Recall by Class', fontsize=13, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # F1-Score
    sns.barplot(data=metrics_df, x='Class', y='F1-Score', ax=axes[2], palette='Oranges')
    axes[2].set_title(f'{title_prefix}F1-Score by Class', fontsize=13, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df

print("\n===== Class-wise Performance Metrics =====")

# Validation class-wise metrics
print("\n[Validation Set - DR]")
val_dr_metrics_df = plot_class_wise_metrics(val_dr_labels, val_dr_preds, dr_class_names, 
                                         "Validation DR - ")
print(val_dr_metrics_df.round(4))

print("\n[Validation Set - DME]")
val_dme_metrics_df = plot_class_wise_metrics(val_dme_labels, val_dme_preds, dme_class_names,
                                          "Validation DME - ")
print(val_dme_metrics_df.round(4))

# Test class-wise metrics
print("\n[Test Set - DR]")
test_dr_metrics_df = plot_class_wise_metrics(test_dr_labels, test_dr_preds, dr_class_names,
                                          "Test DR - ")
print(test_dr_metrics_df.round(4))

print("\n[Test Set - DME]")
test_dme_metrics_df = plot_class_wise_metrics(test_dme_labels, test_dme_preds, dme_class_names,
                                           "Test DME - ")
print(test_dme_metrics_df.round(4))

# %%
# Summary Comparison: Validation vs Test
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Validation vs Test Set Performance Comparison', fontsize=16, fontweight='bold')

# DR Accuracy comparison
dr_metrics_comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Macro'],
    'Validation': [val_acc_dr, val_f1_dr],
    'Test': [test_acc_dr, test_f1_dr]
})

axes[0, 0].bar(dr_metrics_comparison['Metric'], dr_metrics_comparison['Validation'], 
               label='Validation', alpha=0.7, color='blue', width=0.35)
axes[0, 0].bar([x + 0.35 for x in range(len(dr_metrics_comparison))], 
               dr_metrics_comparison['Test'], label='Test', alpha=0.7, color='orange', width=0.35)
axes[0, 0].set_title('DR Performance Comparison', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Score', fontsize=12)
axes[0, 0].set_ylim([0, 1])
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_xticks([x + 0.175 for x in range(len(dr_metrics_comparison))])
axes[0, 0].set_xticklabels(dr_metrics_comparison['Metric'])

# DME Accuracy comparison
dme_metrics_comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Macro'],
    'Validation': [val_acc_dme, val_f1_dme],
    'Test': [test_acc_dme, test_f1_dme]
})

axes[0, 1].bar(dme_metrics_comparison['Metric'], dme_metrics_comparison['Validation'], 
               label='Validation', alpha=0.7, color='blue', width=0.35)
axes[0, 1].bar([x + 0.35 for x in range(len(dme_metrics_comparison))], 
               dme_metrics_comparison['Test'], label='Test', alpha=0.7, color='orange', width=0.35)
axes[0, 1].set_title('DME Performance Comparison', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Score', fontsize=12)
axes[0, 1].set_ylim([0, 1])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_xticks([x + 0.175 for x in range(len(dme_metrics_comparison))])
axes[0, 1].set_xticklabels(dme_metrics_comparison['Metric'])

# DR F1 by class
dr_f1_comparison = pd.DataFrame({
    'Class': dr_class_names,
    'Validation': val_dr_metrics_df['F1-Score'].values,
    'Test': test_dr_metrics_df['F1-Score'].values
})
x_pos = np.arange(len(dr_f1_comparison))
axes[1, 0].bar(x_pos - 0.2, dr_f1_comparison['Validation'], 0.4, label='Validation', 
               alpha=0.7, color='blue')
axes[1, 0].bar(x_pos + 0.2, dr_f1_comparison['Test'], 0.4, label='Test', 
               alpha=0.7, color='orange')
axes[1, 0].set_title('DR F1-Score by Class', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('F1-Score', fontsize=12)
axes[1, 0].set_xlabel('Class', fontsize=12)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(dr_class_names, rotation=45, ha='right')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# DME F1 by class
dme_f1_comparison = pd.DataFrame({
    'Class': dme_class_names,
    'Validation': val_dme_metrics_df['F1-Score'].values,
    'Test': test_dme_metrics_df['F1-Score'].values
})
x_pos = np.arange(len(dme_f1_comparison))
axes[1, 1].bar(x_pos - 0.2, dme_f1_comparison['Validation'], 0.4, label='Validation', 
               alpha=0.7, color='blue')
axes[1, 1].bar(x_pos + 0.2, dme_f1_comparison['Test'], 0.4, label='Test', 
               alpha=0.7, color='orange')
axes[1, 1].set_title('DME F1-Score by Class', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('F1-Score', fontsize=12)
axes[1, 1].set_xlabel('Class', fontsize=12)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(dme_class_names, rotation=45, ha='right')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RUN_DIR / 'validation_vs_test_comparison.png', dpi=300, bbox_inches='tight')
print(f"[Info] Comparison plot saved to {RUN_DIR / 'validation_vs_test_comparison.png'}")
plt.show()

# %%
CAL_CKPT_PATH = Path("checkpoints/best_idrid_simple_calibrated.pt")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "temp_dr_state_dict": temp_dr.state_dict(),
        "temp_dme_state_dict": temp_dme.state_dict(),
    },
    CAL_CKPT_PATH
)
print(f"[Info] Saved calibrated model + temperatures to {CAL_CKPT_PATH}")


# %%
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
        }
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
        }
    }
}

RUN_DIR.mkdir(parents=True, exist_ok=True)
with open(RUN_DIR / "metrics_simple.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"[Info] Metrics saved to {RUN_DIR / 'metrics_simple.json'}")
print("[Done] Training + evaluation complete.")



