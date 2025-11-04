# -*- coding: utf-8 -*-
import os
import shutil
import csv
from typing import Sequence
from sklearn.model_selection import train_test_split
import numpy as np

# =========================
# Parameter Settings (Please confirm paths)
# =========================
BASE_FOLDER = '/home/tonyliao/WIFI_SENSING_LOCATION'   # Fixed base folder
CLASSES = ['Empty_FineTune', 'Stationary_FineTune']                      # Binary classification
FILE_EXTENSIONS: Sequence[str] = ('.npz', '.npy', '.jpg', '.jpeg', '.png')  # Supports CSI and image files

TEST_RATIO = 0.20       # Proportion for test set
VAL_RATIO = 0.20        # Proportion for validation set from remaining data
RANDOM_SEED = 42

# Whether to compute AveCSI (only for CSI files: .npz/.npy)
COMPUTE_AVECSI = True
AVECSI_OUTPUT = os.path.join(BASE_FOLDER, 'avecsi_empty.npz')
FIXED_CROP_LENGTH = None   # e.g., 512; None means no fixed length

# =========================
# Tools
# =========================
def is_valid_file(path: str) -> bool:
    return path.lower().endswith(FILE_EXTENSIONS)

def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)

def load_csi(path: str) -> np.ndarray:
    """Load .npz/.npy, expecting array 'X' or first array; output shape [C, T]."""
    if path.endswith('.npz'):
        z = np.load(path)
        x = z['X'] if 'X' in z.files else z[z.files[0]]
    else:
        x = np.load(path)
    x = x.astype(np.float32)
    if x.ndim == 2 and x.shape[1] < x.shape[0]:
        x = x.T
    elif x.ndim == 1:
        x = x[None, :]
    return x

def compute_avecsi_from_files(files: list[str], crop_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu[C,1], sigma[C,1]) averaged from empty room samples."""
    mu_list, sd_list = [], []
    for p in files:
        x = load_csi(p)  # [C, T]
        if crop_len is not None and x.shape[1] > crop_len:
            start = np.random.randint(0, x.shape[1] - crop_len + 1)
            x = x[:, start:start+crop_len]
        mu_list.append(x.mean(axis=1, keepdims=True))           # [C,1]
        sd_list.append(x.std(axis=1, keepdims=True) + 1e-6)     # [C,1]
    mu = np.mean(np.stack(mu_list, axis=0), axis=0)             # [C,1]
    sigma = np.mean(np.stack(sd_list, axis=0), axis=0)          # [C,1]
    return mu, sigma

# =========================
# Exception Handling
# =========================
if not os.path.isdir(BASE_FOLDER):
    raise FileNotFoundError(f"Folder not found: {BASE_FOLDER}")

print("Using base folder:", BASE_FOLDER)

sample_paths, labels = [], []
for c in CLASSES:
    class_folder = os.path.join(BASE_FOLDER, c)
    if not os.path.isdir(class_folder):
        print(f"Warning: Class folder not found {class_folder}, skipping.")
        continue
    for fname in os.listdir(class_folder):
        src = os.path.join(class_folder, fname)
        if os.path.isfile(src) and is_valid_file(src):
            sample_paths.append(src)
            labels.append(c)

if len(sample_paths) == 0:
    raise RuntimeError("No valid files found. Please check class folders and file extensions.")

# =========================
# Train / Val / Test split
# =========================
trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
    sample_paths, labels, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=labels
)

val_ratio_adjusted = VAL_RATIO / (1.0 - TEST_RATIO)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    trainval_paths, trainval_labels, test_size=val_ratio_adjusted, random_state=RANDOM_SEED, stratify=trainval_labels
)

print(f"Total: {len(sample_paths)} | Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

# =========================
# Create folders and copy files
# =========================
train_root = os.path.join(BASE_FOLDER, 'training_set')
val_root   = os.path.join(BASE_FOLDER, 'val_set')
test_root  = os.path.join(BASE_FOLDER, 'test_set')

for root in (train_root, val_root, test_root):
    for c in CLASSES:
        ensure_folder(os.path.join(root, c))

def copy_to_destination(paths: list[str], labels: list[str], dest_root: str):
    for p, y in zip(paths, labels):
        dst = os.path.join(dest_root, y, os.path.basename(p))
        ensure_folder(os.path.dirname(dst))
        shutil.copy2(p, dst)

copy_to_destination(train_paths, train_labels, train_root)
copy_to_destination(val_paths,   val_labels,   val_root)
copy_to_destination(test_paths,  test_labels,  test_root)

print("Train / Val / Test file distribution completed!")

# =========================
# Save split list CSV
# =========================
split_csv = os.path.join(BASE_FOLDER, 'split_list.csv')
with open(split_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Split', 'Label', 'File Path'])
    for p,y in zip(train_paths, train_labels): w.writerow(['train', y, p])
    for p,y in zip(val_paths,   val_labels):   w.writerow(['val',   y, p])
    for p,y in zip(test_paths,  test_labels):  w.writerow(['test',  y, p])
print("Split list saved:", split_csv)

# =========================
# Compute AveCSI
# =========================
if COMPUTE_AVECSI:
    empty_train_dir = os.path.join(train_root, 'empty')
    if os.path.isdir(empty_train_dir):
        empty_files = [os.path.join(empty_train_dir, f) for f in os.listdir(empty_train_dir)
                       if f.lower().endswith(('.npz', '.npy'))]
        if len(empty_files) == 0:
            print("[AveCSI] Skipped: No CSI files (.npz/.npy) found in training_set/empty.")
        else:
            mu, sigma = compute_avecsi_from_files(empty_files, crop_len=FIXED_CROP_LENGTH)
            np.savez(AVECSI_OUTPUT, mu=mu, sigma=sigma)
            print(f"[AveCSI] Saved: {AVECSI_OUTPUT} | mu shape: {mu.shape}  sigma shape: {sigma.shape}")
    else:
        print("[AveCSI] Skipped: training_set/empty folder not found.")