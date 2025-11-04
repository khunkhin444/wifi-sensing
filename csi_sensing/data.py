"""Data loading utilities for CSI sensing."""
from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NormalizationStats:
    """Per-subcarrier normalization statistics."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, window: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to a CSI window."""
        std = np.where(self.std == 0, 1.0, self.std)
        return (window - self.mean) / std

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, Iterable[float]]) -> "NormalizationStats":
        return cls(mean=np.asarray(list(data["mean"]), dtype=np.float32), std=np.asarray(list(data["std"]), dtype=np.float32))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "NormalizationStats":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return cls.from_dict(data)
        raise ValueError(f"Unrecognised scaler format at {path}")


class CSIDataset(Dataset):
    """Dataset for CSI windows described by a CSV manifest."""

    def __init__(
        self,
        manifest,
        data_root: str,
        normalization: Optional[NormalizationStats] = None,
        max_range_cm: float = 1000.0,
        denoise: bool = False,
        denoise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            manifest: Pandas DataFrame or sequence of dict-like rows with keys
                ``path``, ``label_empty``, and ``distA_cm``.
            data_root: Root directory containing CSI window files.
            normalization: Normalization statistics. If ``None`` no normalization is applied.
            max_range_cm: Maximum range used when environment is empty.
            denoise: Whether to apply an optional denoising filter.
            denoise_fn: Callable used when ``denoise`` is True. If ``None`` and ``denoise`` is True,
                a Savitzky–Golay filter is attempted.
        """

        self.manifest = list(manifest)
        self.data_root = data_root
        self.normalization = normalization
        self.max_range_cm = float(max_range_cm)
        self.denoise = denoise
        self._denoise_fn = denoise_fn

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.manifest[index]
        if isinstance(row, dict):
            path = row["path"]
            label_empty = row["label_empty"]
            dist = row["distA_cm"]
        else:  # pandas Series
            path = row.path
            label_empty = row.label_empty
            dist = row.distA_cm

        window = self._load_window(os.path.join(self.data_root, path))

        if self.denoise:
            window = self._apply_denoise(window)

        if self.normalization is not None:
            window = self.normalization.apply(window)

        if window.ndim == 2:
            window = window[None, ...]
        elif window.ndim == 3 and window.shape[0] != 1:
            # collapse channel dimension if provided differently
            window = window.mean(axis=0, keepdims=True)

        window = torch.from_numpy(window.astype(np.float32))
        label_tensor = torch.tensor(float(label_empty), dtype=torch.float32)
        dist_tensor = torch.tensor(float(dist), dtype=torch.float32)
        return {
            "window": window,
            "label_empty": label_tensor,
            "distA_cm": dist_tensor,
        }

    def _load_window(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path)
        elif ext == ".pt":
            arr = torch.load(path, map_location="cpu")
            arr = arr.cpu().numpy()
        else:
            raise ValueError(f"Unsupported window format: {ext}")
        return np.asarray(arr, dtype=np.float32)

    def _apply_denoise(self, window: np.ndarray) -> np.ndarray:
        fn = self._denoise_fn or _default_savgol
        return fn(window)


def _default_savgol(window: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scipy is required for Savitzky–Golay denoising") from exc

    if window.ndim == 3 and window.shape[0] == 1:
        window_2d = window[0]
    else:
        window_2d = window

    polyorder = 2
    win_length = min(9, window_2d.shape[0] // 2 * 2 + 1)
    if win_length < polyorder + 2:
        return window

    filtered = savgol_filter(window_2d, window_length=win_length, polyorder=polyorder, axis=0)
    if window.ndim == 3:
        return filtered[None, ...]
    return filtered


def compute_normalization_stats(dataset: Dataset) -> NormalizationStats:
    """Compute per-subcarrier normalization statistics from a dataset."""
    sums = None
    sq_sums = None
    count = 0

    for item in dataset:
        window = item["window"].numpy()
        if window.ndim == 3 and window.shape[0] == 1:
            window = window[0]
        elif window.ndim == 3:
            window = window.mean(axis=0)

        if sums is None:
            sums = np.zeros(window.shape[1], dtype=np.float64)
            sq_sums = np.zeros_like(sums)

        sums += window.sum(axis=0)
        sq_sums += np.square(window).sum(axis=0)
        count += window.shape[0]

    if count == 0:
        raise ValueError("Dataset is empty; cannot compute normalization stats")

    mean = sums / count
    var = sq_sums / count - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-6))
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def stratified_split(df, train_ratio: float, val_ratio: float, test_ratio: float, random_state: int = 0):
    """Stratified split of a pandas DataFrame."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("Ratios must sum to 1.0")

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    train_df, temp_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df["label_empty"],
        random_state=random_state,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        stratify=temp_df["label_empty"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
