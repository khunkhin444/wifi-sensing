# -*- coding: utf-8 -*-
import os
import re
import csv
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# Parameter Settings (Please confirm paths)
# =========================
BASE_FOLDER = '/home/tonyliao/WIFI_SENSING_LOCATION'   # Fixed base folder
CLASSES = ['Empty', 'Stationary']                      # Binary classification (display names)
CLASS_SOURCE_FOLDERS = {
    'Empty': 'Empty_FineTune',
    'Stationary': 'Stationary_FineTune',
}  # Map display names -> source folders exported from MATLAB
FILE_EXTENSIONS: Sequence[str] = ('.npz', '.npy', '.jpg', '.jpeg', '.png', '.mat')  # Includes MATLAB metadata

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
if CLASS_SOURCE_FOLDERS:
    print("Class source folders:", CLASS_SOURCE_FOLDERS)


@dataclass(frozen=True)
class SampleGroup:
    """Container for all exported files that share the same numeric index."""

    class_name: str
    index: str
    files: tuple[str, ...]


GROUP_PATTERN = re.compile(r"^(?P<prefix>.+?)_(?P<index>\d+)(?P<ext>\.[^.]+)$")


def iter_class_groups(base_folder: str, class_name: str, source_folder: str | None = None) -> list[SampleGroup]:
    folder_name = source_folder or class_name
    class_folder = os.path.join(base_folder, folder_name)
    if not os.path.isdir(class_folder):
        print(f"Warning: Class folder not found {class_folder} (class '{class_name}'), skipping.")
        return []

    grouped: defaultdict[str, list[str]] = defaultdict(list)
    skipped_files: list[str] = []

    for fname in sorted(os.listdir(class_folder)):
        src = os.path.join(class_folder, fname)
        if not os.path.isfile(src) or not is_valid_file(src):
            continue
        match = GROUP_PATTERN.match(fname)
        if not match:
            skipped_files.append(fname)
            continue
        index = match.group('index')
        grouped[index].append(src)

    if skipped_files:
        print(f"[Info] {class_name}: Skipped {len(skipped_files)} file(s) without numeric suffix.")

    sample_groups: list[SampleGroup] = []
    for index in sorted(grouped.keys(), key=lambda s: int(s)):
        files = tuple(sorted(grouped[index]))
        sample_groups.append(SampleGroup(class_name=class_name, index=index, files=files))

    return sample_groups


sample_groups: list[SampleGroup] = []
labels: list[str] = []
for cls in CLASSES:
    groups = iter_class_groups(BASE_FOLDER, cls, CLASS_SOURCE_FOLDERS.get(cls))
    sample_groups.extend(groups)
    labels.extend([cls] * len(groups))

if len(sample_groups) == 0:
    raise RuntimeError("No valid groups found. Please ensure exported MATLAB windows exist and filenames contain numeric suffixes.")

# =========================
# Train / Val / Test split
# =========================
trainval_groups, test_groups, trainval_labels, test_labels = train_test_split(
    sample_groups,
    labels,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED,
    stratify=labels,
)

val_ratio_adjusted = VAL_RATIO / (1.0 - TEST_RATIO)
train_groups, val_groups, train_labels, val_labels = train_test_split(
    trainval_groups,
    trainval_labels,
    test_size=val_ratio_adjusted,
    random_state=RANDOM_SEED,
    stratify=trainval_labels,
)

print(
    "Total groups: {total} | Train: {train} | Val: {val} | Test: {test}".format(
        total=len(sample_groups),
        train=len(train_groups),
        val=len(val_groups),
        test=len(test_groups),
    )
)

# =========================
# Create folders and copy files
# =========================
train_root = os.path.join(BASE_FOLDER, 'training_set')
val_root   = os.path.join(BASE_FOLDER, 'val_set')
test_root  = os.path.join(BASE_FOLDER, 'test_set')

for root in (train_root, val_root, test_root):
    for c in CLASSES:
        ensure_folder(os.path.join(root, c))


def numbering_width_map(split_groups: dict[str, list[SampleGroup]]) -> dict[tuple[str, str], int]:
    widths: dict[tuple[str, str], int] = {}
    for split, groups in split_groups.items():
        per_class = Counter(g.class_name for g in groups)
        for cls, count in per_class.items():
            widths[(split, cls)] = max(2, len(str(count)))
    return widths


def format_with_new_index(filename: str, new_index: int, width: int) -> str:
    match = GROUP_PATTERN.match(filename)
    if not match:
        return filename
    prefix = match.group('prefix')
    ext = match.group('ext')
    return f"{prefix}_{new_index:0{width}d}{ext}"


def copy_groups_to_destination(
    groups: list[SampleGroup],
    dest_root: str,
    split_name: str,
    widths: dict[tuple[str, str], int],
) -> list[tuple[str, str, str]]:
    """Copy grouped files into destination split, renumbering sequentially.

    Returns a list of tuples (split, label, destination_path) for CSV logging.
    """

    counter: Counter = Counter()
    csv_rows: list[tuple[str, str, str]] = []

    for group in groups:
        key = (split_name, group.class_name)
        counter[key] += 1
        seq_num = counter[key]
        width = widths.get(key, max(2, len(str(seq_num))))

        for src in group.files:
            dst_name = format_with_new_index(os.path.basename(src), seq_num, width)
            dst_path = os.path.join(dest_root, group.class_name, dst_name)
            ensure_folder(os.path.dirname(dst_path))
            shutil.copy2(src, dst_path)
            csv_rows.append((split_name, group.class_name, dst_path))

    return csv_rows


split_groups = {'train': train_groups, 'val': val_groups, 'test': test_groups}
width_map = numbering_width_map(split_groups)

csv_records: list[tuple[str, str, str]] = []
csv_records.extend(copy_groups_to_destination(train_groups, train_root, 'train', width_map))
csv_records.extend(copy_groups_to_destination(val_groups, val_root, 'val', width_map))
csv_records.extend(copy_groups_to_destination(test_groups, test_root, 'test', width_map))

print("Train / Val / Test group distribution completed!")

# =========================
# Save split list CSV
# =========================
split_csv = os.path.join(BASE_FOLDER, 'split_list.csv')
with open(split_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Split', 'Label', 'File Path'])
    for split, label, path in csv_records:
        writer.writerow([split, label, path])
print("Split list saved:", split_csv)

# =========================
# Compute AveCSI
# =========================
if COMPUTE_AVECSI:
    empty_class = next((c for c in CLASSES if 'empty' in c.lower()), None)
    if empty_class is None:
        print("[AveCSI] Skipped: No class contains 'empty'.")
    else:
        empty_train_dir = os.path.join(train_root, empty_class)
        if os.path.isdir(empty_train_dir):
            def is_amplitude_file(name: str) -> bool:
                lower = name.lower()
                if not lower.endswith(('.npz', '.npy')):
                    return False
                if 'phase' in lower or 'feat' in lower or 'sincos' in lower:
                    return False
                return True

            empty_files = [
                os.path.join(empty_train_dir, f)
                for f in os.listdir(empty_train_dir)
                if is_amplitude_file(f)
            ]
            if len(empty_files) == 0:
                print("[AveCSI] Skipped: No amplitude CSI files found in training set for empty class.")
            else:
                mu, sigma = compute_avecsi_from_files(empty_files, crop_len=FIXED_CROP_LENGTH)
                np.savez(AVECSI_OUTPUT, mu=mu, sigma=sigma)
                print(
                    f"[AveCSI] Saved: {AVECSI_OUTPUT} | mu shape: {mu.shape}  sigma shape: {sigma.shape}"
                )
        else:
            print(f"[AveCSI] Skipped: training_set/{empty_class} folder not found.")
