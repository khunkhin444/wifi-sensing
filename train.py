#!/usr/bin/env python
"""Training entry-point for CSI-based sensing."""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from csi_sensing.data import CSIDataset, compute_normalization_stats, stratified_split
from csi_sensing.evaluation import evaluate_model
from csi_sensing.model import CSISensingModel
from csi_sensing.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CSI sensing model")
    parser.add_argument("--data_root", required=True, help="Root directory containing CSI windows")
    parser.add_argument("--csv", required=True, help="CSV manifest with columns path,label_empty,distA_cm")
    parser.add_argument("--out_dir", required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_dist", type=float, default=0.5)
    parser.add_argument("--sigma_cm", type=float, default=200.0)
    parser.add_argument("--max_range_cm", type=float, default=1000.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoise", action="store_true", help="Apply optional Savitzky-Golay denoising")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    return {
        "window": torch.stack([item["window"] for item in batch]),
        "label_empty": torch.stack([item["label_empty"] for item in batch]),
        "distA_cm": torch.stack([item["distA_cm"] for item in batch]),
    }


def train_one_epoch(model, dataloader, optimizer, device, scaler, lambda_dist, clip_grad, amp):
    model.train()
    bce_loss = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss(reduction="none")
    total_loss = 0.0
    total_empty = 0.0
    total_dist = 0.0
    num_batches = 0

    for batch in dataloader:
        inputs = batch["window"].to(device)
        labels = batch["label_empty"].to(device)
        dists = batch["distA_cm"].to(device)
        mask_present = (labels == 0).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits, dist_pred = model(inputs)
            loss_empty = bce_loss(logits, labels)
            dist_loss = smooth_l1(dist_pred, dists)
            if mask_present.sum() > 0:
                dist_loss = (dist_loss * mask_present).sum() / mask_present.sum()
            else:
                dist_loss = torch.tensor(0.0, device=device)
            loss = loss_empty + lambda_dist * dist_loss

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()
        total_empty += loss_empty.item()
        total_dist += dist_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "loss_empty": total_empty / num_batches,
        "loss_dist": total_dist / num_batches,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.csv)
    if not {"path", "label_empty", "distA_cm"}.issubset(df.columns):
        raise ValueError("CSV must contain path,label_empty,distA_cm columns")

    train_df, val_df, test_df = stratified_split(df, args.train_ratio, args.val_ratio, args.test_ratio, random_state=args.seed)

    # Compute normalization stats on raw training dataset
    train_dataset_raw = CSIDataset(train_df.itertuples(index=False), args.data_root, normalization=None, max_range_cm=args.max_range_cm, denoise=args.denoise)
    stats = compute_normalization_stats(train_dataset_raw)
    stats_path = os.path.join(args.out_dir, "scaler.pkl")
    stats.save(stats_path)

    train_dataset = CSIDataset(train_df.itertuples(index=False), args.data_root, stats, args.max_range_cm, denoise=args.denoise)
    val_dataset = CSIDataset(val_df.itertuples(index=False), args.data_root, stats, args.max_range_cm, denoise=args.denoise)
    test_dataset = CSIDataset(test_df.itertuples(index=False), args.data_root, stats, args.max_range_cm, denoise=args.denoise)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = CSISensingModel()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    best_f1 = -float("inf")
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler, args.lambda_dist, args.clip_grad, args.amp)
        val_metrics = evaluate_model(model, val_loader, device, args.sigma_cm, threshold_empty=0.5)

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val_presence_f1": val_metrics["presence"].f1,
        })

        f1 = val_metrics["presence"].f1
        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, args, stats, epoch, os.path.join(args.out_dir, "best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    # Load best checkpoint for evaluation
    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    metrics_val = evaluate_model(model, val_loader, device, args.sigma_cm, threshold_empty=0.5)
    metrics_test = evaluate_model(model, test_loader, device, args.sigma_cm, threshold_empty=0.5)

    metrics_payload: Dict[str, Dict] = {
        "val": serialize_metrics(metrics_val),
        "test": serialize_metrics(metrics_test),
    }

    save_json(metrics_payload, os.path.join(args.out_dir, "metrics_val.json"))
    save_history(history, os.path.join(args.out_dir, "history.json"))
    create_plots(metrics_val, args.out_dir, prefix="val")
    create_plots(metrics_test, args.out_dir, prefix="test")

    # Persist split assignments for downstream evaluation
    combined = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ])
    combined.to_csv(os.path.join(args.out_dir, "splits.csv"), index=False)


def save_checkpoint(model, optimizer, args, stats, epoch, path: str) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": vars(args),
        "normalization": stats.to_dict(),
    }, path)


def serialize_metrics(metrics_dict: Dict) -> Dict:
    def _convert(mapping: Dict) -> Dict:
        converted = {}
        for key, value in mapping.items():
            if isinstance(value, (np.floating, np.integer)):
                converted[key] = float(value)
            else:
                converted[key] = value
        return converted

    return {
        "presence": _convert(metrics_dict["presence"].__dict__),
        "calibration": _convert(metrics_dict["calibration"].__dict__),
        "distance": _convert(metrics_dict["distance"].__dict__),
        "mean_pA": float(metrics_dict["mean_pA"]),
        "corr_proximity_pA": None if metrics_dict["corr_proximity_pA"] is None else float(metrics_dict["corr_proximity_pA"]),
    }


def save_history(history, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def create_plots(metrics_dict: Dict, out_dir: str, prefix: str) -> None:
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)

    labels = metrics_dict["labels"]
    probs = metrics_dict["probs_empty"]
    dist_true = metrics_dict["dist_true"]
    dist_pred = metrics_dict["dist_pred"]
    errors = np.abs(dist_pred[labels == 0] - dist_true[labels == 0])

    if len(np.unique(labels)) > 1:
        from sklearn.metrics import roc_curve, precision_recall_curve

        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{prefix.upper()} ROC")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"))
        plt.close()

        precision, recall, _ = precision_recall_curve(labels, probs)
        plt.figure()
        plt.plot(recall, precision, label="PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{prefix.upper()} PR")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png"))
        plt.close()

    if errors.size > 0:
        sorted_errors = np.sort(errors)
        cdf = np.linspace(0, 1, len(sorted_errors))
        plt.figure()
        plt.plot(sorted_errors, cdf)
        plt.xlabel("|Error| (cm)")
        plt.ylabel("CDF")
        plt.title(f"{prefix.upper()} Distance Error CDF")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{prefix}_error_cdf.png"))
        plt.close()


if __name__ == "__main__":
    main()
