#!/usr/bin/env python
"""Inference script for CSI sensing."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from csi_sensing.data import NormalizationStats
from csi_sensing.model import CSISensingModel
from csi_sensing.utils import compute_location_a_percentage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a CSI window")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--npy_window", required=True, help="Path to .npy or .pt window file")
    parser.add_argument("--sigma_cm", type=float, default=200.0)
    parser.add_argument("--threshold_empty", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_window(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        window = np.load(path)
    elif ext == ".pt":
        window = torch.load(path, map_location="cpu").numpy()
    else:
        raise ValueError("Unsupported window extension")
    return np.asarray(window, dtype=np.float32)


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    normalization = None
    if "normalization" in ckpt:
        normalization = NormalizationStats.from_dict(ckpt["normalization"])
    else:
        scaler_path = os.path.join(os.path.dirname(args.ckpt), "scaler.pkl")
        if os.path.exists(scaler_path):
            normalization = NormalizationStats.load(scaler_path)

    window = load_window(args.npy_window)
    if window.ndim == 2:
        window = window[None, ...]
    elif window.ndim == 3 and window.shape[0] != 1:
        window = window.mean(axis=0, keepdims=True)

    if normalization is not None:
        window = normalization.apply(window[0])[None, ...]

    tensor = torch.from_numpy(window).unsqueeze(0).to(torch.device(args.device))

    model = CSISensingModel()
    model.load_state_dict(ckpt["model_state"])
    model.to(torch.device(args.device))
    model.eval()

    with torch.no_grad():
        logits, dist_pred = model(tensor)
        probs_empty = torch.sigmoid(logits).item()
        dist_cm = dist_pred.item()

    p_a = compute_location_a_percentage(np.array([dist_cm]), args.sigma_cm, np.array([probs_empty]), args.threshold_empty)[0]

    output = {
        "p_empty": float(probs_empty),
        "dA_cm": float(dist_cm),
        "pA": float(p_a),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
