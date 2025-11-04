"""Generate a synthetic dataset for smoke-testing the CSI sensing pipeline."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dummy CSI dataset")
    parser.add_argument("--out_dir", required=True, help="Directory to store generated data")
    parser.add_argument("--num_samples", type=int, default=120)
    parser.add_argument("--time_steps", type=int, default=32)
    parser.add_argument("--subcarriers", type=int, default=16)
    parser.add_argument("--max_range_cm", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(args.num_samples):
        label_empty = int(idx % 3 == 0)
        if label_empty:
            dist_cm = args.max_range_cm
        else:
            dist_cm = float((idx % 10 + 1) * 50.0)

        base_signal = rng.normal(size=(args.time_steps, args.subcarriers)).astype("float32")
        trend = np.linspace(0, 1, args.time_steps, dtype="float32")[:, None]
        window = base_signal + (1 - label_empty) * trend * (1.0 / (dist_cm / 50.0))

        path = f"sample_{idx:04d}.npy"
        np.save(windows_dir / path, window)
        rows.append({"path": os.path.join("windows", path), "label_empty": label_empty, "distA_cm": dist_cm})

    csv_path = out_dir / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label_empty", "distA_cm"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} samples at {csv_path}")


if __name__ == "__main__":
    main()
