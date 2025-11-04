#!/usr/bin/env python
"""Evaluation script for CSI sensing."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from csi_sensing.data import CSIDataset, NormalizationStats
from csi_sensing.evaluation import evaluate_model
from csi_sensing.model import CSISensingModel
from csi_sensing.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CSI sensing model")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--csv", required=True, help="CSV manifest")
    parser.add_argument("--data_root", required=True, help="Root directory for windows")
    parser.add_argument("--out_dir", required=True, help="Directory to store metrics and plots")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sigma_cm", type=float, default=200.0)
    parser.add_argument("--threshold_empty", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    return parser.parse_args()


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError("Checkpoint missing model_state")
    return ckpt


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    df = pd.read_csv(args.csv)
    ckpt = load_checkpoint(args.ckpt)

    normalization = NormalizationStats.from_dict(ckpt.get("normalization")) if "normalization" in ckpt else None
    if normalization is None:
        scaler_path = os.path.join(os.path.dirname(args.ckpt), "scaler.pkl")
        normalization = NormalizationStats.load(scaler_path)

    if args.split == "all" or "split" not in df.columns:
        eval_df = df
        if args.split != "all" and "split" not in df.columns:
            print("Warning: split column missing; evaluating on entire manifest.")
    else:
        eval_df = df[df["split"] == args.split]
        if eval_df.empty:
            raise ValueError(f"No samples found for split {args.split}")

    dataset = CSIDataset(eval_df.itertuples(index=False), args.data_root, normalization)

    def collate_fn(batch):
        return {
            "window": torch.stack([item["window"] for item in batch]),
            "label_empty": torch.stack([item["label_empty"] for item in batch]),
            "distA_cm": torch.stack([item["distA_cm"] for item in batch]),
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model = CSISensingModel()
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model.to(device)

    metrics_dict = evaluate_model(model, dataloader, device, args.sigma_cm, args.threshold_empty)
    serialized = serialize_metrics(metrics_dict)
    save_json(serialized, os.path.join(args.out_dir, "metrics.json"))
    create_plots(metrics_dict, args.out_dir)

    print(json.dumps(serialized, indent=2))


def serialize_metrics(metrics_dict):
    def _convert(mapping):
        converted = {}
        for key, value in mapping.items():
            if isinstance(value, (np.floating, np.integer)):
                converted[key] = float(value)
            else:
                converted[key] = value
        return converted

    return {
        "presence": _convert(metrics_dict["presence"].__dict__),
        "distance": _convert(metrics_dict["distance"].__dict__),
        "calibration": _convert(metrics_dict["calibration"].__dict__),
        "mean_pA": float(metrics_dict["mean_pA"]),
        "corr_proximity_pA": None if metrics_dict["corr_proximity_pA"] is None else float(metrics_dict["corr_proximity_pA"]),
    }


def create_plots(metrics_dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)

    labels = metrics_dict["labels"]
    probs = metrics_dict["probs_empty"]

    if len(np.unique(labels)) > 1:
        from sklearn.metrics import roc_curve, precision_recall_curve

        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "roc.png"))
        plt.close()

        precision, recall, _ = precision_recall_curve(labels, probs)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "pr.png"))
        plt.close()

    mask = labels == 0
    errors = np.abs(metrics_dict["dist_pred"][mask] - metrics_dict["dist_true"][mask])
    if errors.size > 0:
        sorted_errors = np.sort(errors)
        cdf = np.linspace(0, 1, len(sorted_errors))
        plt.figure()
        plt.plot(sorted_errors, cdf)
        plt.xlabel("|Error| (cm)")
        plt.ylabel("CDF")
        plt.title("Distance Error CDF")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "error_cdf.png"))
        plt.close()


if __name__ == "__main__":
    main()
