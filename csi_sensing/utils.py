"""Utility functions for CSI sensing."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics


@dataclass
class PresenceMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: Optional[float]
    auprc: Optional[float]
    brier: float


@dataclass
class DistanceMetrics:
    mae: float
    rmse: float
    median: float


@dataclass
class CalibrationMetrics:
    ece: float
    brier: float


def compute_location_a_percentage(distance_cm: np.ndarray, sigma_cm: float, empty_prob: Optional[np.ndarray] = None, threshold_empty: float = 0.5) -> np.ndarray:
    """Map distances to a soft percentage toward Location A.

    Args:
        distance_cm: Predicted distances in centimetres.
        sigma_cm: Gaussian kernel width.
        empty_prob: Optional probability that the environment is empty.
        threshold_empty: When ``empty_prob`` exceeds this threshold, ``pA`` is forced to zero.
    """

    sigma_sq = sigma_cm ** 2
    sigma_sq = max(sigma_sq, 1e-6)
    p_a = np.exp(-np.square(distance_cm) / (2.0 * sigma_sq))
    if empty_prob is not None:
        mask = empty_prob > threshold_empty
        p_a = np.where(mask, 0.0, p_a)
    return np.clip(p_a, 0.0, 1.0)


def compute_presence_metrics(labels: np.ndarray, logits: np.ndarray) -> PresenceMetrics:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.int32)

    accuracy = (preds == labels).mean()
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds, zero_division=0)
    f1 = metrics.f1_score(labels, preds, zero_division=0)

    try:
        auroc = metrics.roc_auc_score(labels, probs)
    except ValueError:
        auroc = None
    try:
        auprc = metrics.average_precision_score(labels, probs)
    except ValueError:
        auprc = None

    brier = metrics.brier_score_loss(labels, probs)
    return PresenceMetrics(accuracy, precision, recall, f1, auroc, auprc, brier)


def compute_distance_metrics(dist_true: np.ndarray, dist_pred: np.ndarray) -> DistanceMetrics:
    if dist_true.size == 0:
        return DistanceMetrics(mae=float("nan"), rmse=float("nan"), median=float("nan"))
    errors = np.abs(dist_pred - dist_true)
    mae = errors.mean()
    rmse = math.sqrt(np.mean(np.square(errors)))
    median = np.median(errors)
    return DistanceMetrics(mae, rmse, median)


def compute_calibration_metrics(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> CalibrationMetrics:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        weight = mask.mean()
        ece += np.abs(conf - acc) * weight

    brier = metrics.brier_score_loss(labels, probs)
    return CalibrationMetrics(ece=float(ece), brier=float(brier))


def save_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def correlation(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size == 0 or y.size == 0:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_error_cdf(errors: np.ndarray, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    if errors.size == 0:
        return np.array([]), np.array([])
    sorted_errors = np.sort(errors)
    probs = np.linspace(0, 1, len(sorted_errors))
    return sorted_errors, probs


def softmin_weights(distances: np.ndarray, tau_cm: float = 100.0) -> np.ndarray:
    scaled = -distances / max(tau_cm, 1e-6)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights
