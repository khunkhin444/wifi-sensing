"""Model evaluation helpers."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import (
    CalibrationMetrics,
    DistanceMetrics,
    PresenceMetrics,
    compute_calibration_metrics,
    compute_distance_metrics,
    compute_location_a_percentage,
    compute_presence_metrics,
    correlation,
)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sigma_cm: float,
    threshold_empty: float,
) -> Dict[str, object]:
    model.eval()
    logits_list = []
    dist_preds = []
    labels = []
    dist_true = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["window"].to(device)
            labels_batch = batch["label_empty"].cpu().numpy().astype(np.int32)
            dist_batch = batch["distA_cm"].cpu().numpy()

            logits, dists = model(inputs)
            logits_list.append(logits.detach().cpu().numpy())
            dist_preds.append(dists.detach().cpu().numpy())
            labels.append(labels_batch)
            dist_true.append(dist_batch)

    logits_np = np.concatenate(logits_list, axis=0)
    dist_pred_np = np.concatenate(dist_preds, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    dist_true_np = np.concatenate(dist_true, axis=0)

    presence_metrics: PresenceMetrics = compute_presence_metrics(labels_np, logits_np)
    probs_empty = 1 / (1 + np.exp(-logits_np))
    calibration: CalibrationMetrics = compute_calibration_metrics(labels_np, probs_empty)

    mask_present = labels_np == 0
    distance_metrics: DistanceMetrics = compute_distance_metrics(dist_true_np[mask_present], dist_pred_np[mask_present])
    errors = dist_pred_np[mask_present] - dist_true_np[mask_present]

    p_a = compute_location_a_percentage(dist_pred_np, sigma_cm, probs_empty, threshold_empty)
    mean_p_a = float(p_a.mean())

    corr = correlation(1.0 - dist_true_np / (dist_true_np.max() + 1e-6), p_a)

    return {
        "presence": presence_metrics,
        "calibration": calibration,
        "distance": distance_metrics,
        "labels": labels_np,
        "probs_empty": probs_empty,
        "dist_pred": dist_pred_np,
        "dist_true": dist_true_np,
        "errors": errors,
        "pA": p_a,
        "mean_pA": mean_p_a,
        "corr_proximity_pA": corr,
    }
