#!/usr/bin/env python3
"""Command-line inference pipeline for AveCSI PreCNN localisation.

This script mirrors the CSI_PreCNN_Inference.ipynb notebook so that you can run
localisation from the shell.  It loads a frozen PreCNN backbone, builds feature
embeddings for reference/query CSI windows, reports anchor distances, and
optionally refreshes batch-normalisation statistics on the new session.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorflow import keras

SUPPORTED_FEATURE_MODES = {"amp", "amp+phase", "amp+sin_cos", "amp+phase+sin_cos"}
EPS = 1e-6


def configure_tensorflow() -> None:
    """Match the session initialisation used by the notebook."""

    tf.keras.backend.clear_session()
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            # Some environments may not allow memory growth toggles.
            pass


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def ensure_channels_first(arr: np.ndarray, force_layout: str = "TC") -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if force_layout == "TC":
        return arr.T
    if force_layout == "CT":
        return arr
    raise ValueError(f"Unknown force_layout '{force_layout}'")


def related_window_path(path: str, stem: str) -> str:
    base_dir, fname = os.path.split(path)
    if not fname.startswith("X_window_"):
        raise ValueError(f"Unexpected window filename: {fname}")
    suffix = fname[len("X_window_"):]
    return os.path.join(base_dir, f"{stem}{suffix}")


def wants_raw_phase(mode: str) -> bool:
    return "phase" in mode and mode != "amp+sin_cos"


def wants_sin_cos(mode: str) -> bool:
    return "sin_cos" in mode


def load_window_components(
    path: str,
    feature_mode: str = "amp",
    force_layout: str = "TC",
) -> Tuple[np.ndarray, List[np.ndarray], List[str], set[str]]:
    amp = ensure_channels_first(np.load(path), force_layout=force_layout)
    extras: List[np.ndarray] = []
    names: List[str] = []
    missing: set[str] = set()

    if feature_mode != "amp":
        if wants_raw_phase(feature_mode):
            phase_path = related_window_path(path, "Xphase_window_")
            if os.path.isfile(phase_path):
                phase = ensure_channels_first(np.load(phase_path), force_layout=force_layout)
                extras.append(phase.astype(np.float32))
                names.append("phase")
            else:
                missing.add("phase")
                extras.append(np.zeros_like(amp, dtype=np.float32))
                names.append("phase")

        if wants_sin_cos(feature_mode):
            sc_path = related_window_path(path, "XphaseSC_window_")
            if os.path.isfile(sc_path):
                sc = np.load(sc_path)
                if sc.ndim != 3 or sc.shape[-1] != 2:
                    raise ValueError(f"{sc_path} expected shape [T, S, 2], got {sc.shape}")
                sin_comp = ensure_channels_first(sc[:, :, 0], force_layout=force_layout)
                cos_comp = ensure_channels_first(sc[:, :, 1], force_layout=force_layout)
                extras.extend([sin_comp.astype(np.float32), cos_comp.astype(np.float32)])
                names.extend(["phase_sin", "phase_cos"])
            else:
                missing.update({"phase_sin", "phase_cos"})
                extras.extend(
                    [
                        np.zeros_like(amp, dtype=np.float32),
                        np.zeros_like(amp, dtype=np.float32),
                    ]
                )
                names.extend(["phase_sin", "phase_cos"])

    return amp.astype(np.float32), extras, names, missing


def center_crop(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.shape[1] <= length:
        return arr
    start = max((arr.shape[1] - length) // 2, 0)
    end = start + length
    return arr[:, start:end]


def pad_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.shape[1] == length:
        return arr
    if arr.shape[1] > length:
        return arr[:, :length]
    pad_width = ((0, 0), (0, length - arr.shape[1]))
    return np.pad(arr, pad_width, mode="constant")


def compute_T_stats(
    paths: Sequence[str],
    force_layout: str = "TC",
    crop_len: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    T_list: List[int] = []
    C_first: Optional[int] = None
    for p in paths:
        if not os.path.isfile(p):
            continue
        X = np.load(p)
        if X.ndim != 2:
            continue
        if force_layout == "TC":
            C, T = X.shape[1], X.shape[0]
        else:
            C, T = X.shape
        if C_first is None:
            C_first = int(C)
        T_list.append(int(T if crop_len is None else crop_len))
    if len(T_list) == 0:
        return C_first, None, None
    T_max = max(T_list) if crop_len is None else crop_len
    T_min = min(T_list) if crop_len is None else crop_len
    return C_first, T_min, T_max


def compute_session_norm_stats(
    paths: Sequence[str],
    feature_mode: str = "amp",
    force_layout: str = "TC",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, object]]:
    amp_sum: Optional[np.ndarray] = None
    amp_sumsq: Optional[np.ndarray] = None
    amp_count: int = 0

    extra_acc: Dict[str, Dict[str, Optional[np.ndarray]]] = defaultdict(
        lambda: {"sum": None, "sumsq": None, "count": 0}
    )
    missing_counter: Counter = Counter()

    for path in paths:
        if not os.path.isfile(path):
            continue
        amp, extras, names, missing = load_window_components(
            path, feature_mode=feature_mode, force_layout=force_layout
        )
        amp_sum = (
            amp.sum(axis=1, keepdims=True)
            if amp_sum is None
            else amp_sum + amp.sum(axis=1, keepdims=True)
        )
        amp_sumsq = (
            (amp**2).sum(axis=1, keepdims=True)
            if amp_sumsq is None
            else amp_sumsq + (amp**2).sum(axis=1, keepdims=True)
        )
        amp_count += amp.shape[1]

        for name, arr in zip(names, extras):
            if name in missing:
                missing_counter[name] += 1
                continue
            stats = extra_acc[name]
            stats["sum"] = (
                arr.sum(axis=1, keepdims=True)
                if stats["sum"] is None
                else stats["sum"] + arr.sum(axis=1, keepdims=True)
            )
            stats["sumsq"] = (
                (arr**2).sum(axis=1, keepdims=True)
                if stats["sumsq"] is None
                else stats["sumsq"] + (arr**2).sum(axis=1, keepdims=True)
            )
            stats["count"] = int(stats["count"]) + arr.shape[1]

    if amp_sum is None or amp_count == 0:
        raise RuntimeError(
            "Unable to compute normalization stats; no valid CSI windows were found."
        )

    mu = amp_sum / float(amp_count)
    var = amp_sumsq / float(amp_count) - mu**2
    sigma = np.sqrt(np.maximum(var, EPS)).astype(np.float32)
    mu = mu.astype(np.float32)

    extra_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, stats in extra_acc.items():
        count = int(stats["count"])
        if count == 0 or stats["sum"] is None or stats["sumsq"] is None:
            continue
        mu_e = stats["sum"] / float(count)
        var_e = stats["sumsq"] / float(count) - mu_e**2
        sigma_e = np.sqrt(np.maximum(var_e, EPS)).astype(np.float32)
        extra_stats[name] = (mu_e.astype(np.float32), sigma_e)

    info = {
        "total_windows": len(paths),
        "missing_features": dict(missing_counter),
        "extra_stats_available": sorted(extra_stats.keys()),
    }
    return mu, sigma, extra_stats, info


def build_feature_tensor(
    paths: Sequence[str],
    mu: Optional[np.ndarray],
    sigma: Optional[np.ndarray],
    feature_mode: str = "amp",
    extra_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    force_layout: str = "TC",
    crop_len: Optional[int] = None,
    pad_to: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    extra_stats = extra_stats or {}
    arrays: List[np.ndarray] = []
    component_order: Optional[List[str]] = None
    missing_counter: Counter = Counter()

    if crop_len is None and pad_to is None:
        raise ValueError("pad_to must be provided when crop_len is None")

    for path in paths:
        if not os.path.isfile(path):
            continue
        amp, extras, names, missing = load_window_components(
            path, feature_mode=feature_mode, force_layout=force_layout
        )
        if component_order is None:
            component_order = ["amp"] + names
        if mu is not None and sigma is not None:
            amp = (amp - mu) / (sigma + EPS)

        parts: List[np.ndarray] = [amp.astype(np.float32)]
        for name, arr in zip(names, extras):
            arr = arr.astype(np.float32)
            if name in missing:
                missing_counter[name] += 1
                arr = np.zeros_like(arr)
            elif name in extra_stats:
                mu_e, sigma_e = extra_stats[name]
                arr = (arr - mu_e) / (sigma_e + EPS)
            else:
                missing_counter[f"stats_missing::{name}"] += 1
            parts.append(arr)

        if crop_len is not None:
            parts = [center_crop(part, crop_len) for part in parts]

        target_pad = pad_to if pad_to is not None else crop_len
        parts = [pad_to_length(part, target_pad) for part in parts]

        arrays.append(np.concatenate(parts, axis=0).astype(np.float32))

    if not arrays:
        raise RuntimeError(
            "No tensors were built; check that the metadata points to existing windows."
        )

    X = np.stack(arrays, axis=0).astype(np.float32)
    info = {
        "component_order": component_order if component_order else ["amp"],
        "missing_features": dict(missing_counter),
        "total_windows": len(arrays),
        "pad_to": pad_to,
        "crop_len": crop_len,
    }
    return X, info


# ---------------------------------------------------------------------------
# Metadata + embedding utilities
# ---------------------------------------------------------------------------


def load_metadata(csv_path: str, root: Optional[str] = None) -> pd.DataFrame:
    if not csv_path:
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    required_cols = {"window_path", "point_id", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata {csv_path} is missing required columns: {sorted(missing)}"
        )
    if root:
        df["window_path"] = df["window_path"].apply(
            lambda p: os.path.join(root, p) if not os.path.isabs(p) else p
        )
    df["window_path"] = df["window_path"].astype(str)
    df = df.sort_values("window_path").reset_index(drop=True)
    missing_files = [p for p in df["window_path"] if not os.path.isfile(p)]
    if missing_files:
        print(
            f"[WARN] {len(missing_files)} metadata paths do not exist; they will be ignored in tensor builds."
        )
    return df


def refresh_batch_norm(model: keras.Model, data: np.ndarray, batch_size: int = 128) -> None:
    bn_layers = [layer for layer in model.layers if isinstance(layer, layers.BatchNormalization)]
    if not bn_layers:
        print("[BN] No BatchNormalization layers detected; skipping refresh.")
        return
    print(f"[BN] Refreshing running statistics on {data.shape[0]} samples...")
    ds = tf.data.Dataset.from_tensor_slices(data.astype(np.float32)).batch(batch_size)
    for batch in ds:
        model(batch, training=True)
    print("[BN] Refresh complete.")


def extract_embeddings(
    model: keras.Model, data: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    if data.size == 0:
        return np.empty((0, model.output_shape[-1]), dtype=np.float32)
    return model.predict(data, batch_size=batch_size, verbose=1)


def build_reference_database(
    meta: pd.DataFrame, embeddings: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if len(meta) != len(embeddings):
        raise ValueError("Metadata and embedding counts do not match.")
    grouped = []
    for point_id, group in meta.groupby("point_id"):
        idx = group.index.to_numpy()
        mean_emb = embeddings[idx].mean(axis=0)
        coords = group[["x", "y"]].mean().to_numpy(dtype=np.float32)
        grouped.append(
            {
                "point_id": point_id,
                "x": float(coords[0]),
                "y": float(coords[1]),
                "count": int(len(group)),
                "embedding": mean_emb.astype(np.float32),
            }
        )
    ref_df = pd.DataFrame(grouped).sort_values("point_id").reset_index(drop=True)
    ref_matrix = np.stack(ref_df["embedding"].to_numpy(), axis=0)
    ref_coords = ref_df[["x", "y"]].to_numpy(dtype=np.float32)
    return ref_df, ref_matrix, ref_coords


def soft_rbf_localization(
    query_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_coords: np.ndarray,
    gamma: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if query_embeddings.size == 0:
        return (
            np.empty((0, ref_coords.shape[1]), dtype=np.float32),
            np.empty((0, ref_embeddings.shape[0]), dtype=np.float32),
        )
    preds = []
    weight_records = []
    for q in query_embeddings:
        diff = ref_embeddings - q[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        weights = np.exp(-gamma * dist2)
        if top_k is not None and 0 < top_k < len(weights):
            idx = np.argpartition(dist2, top_k)[:top_k]
            mask = np.zeros_like(weights)
            mask[idx] = 1.0
            weights *= mask
        total = weights.sum()
        if total <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= total
        preds.append(weights @ ref_coords)
        weight_records.append(weights)
    return (
        np.stack(preds, axis=0).astype(np.float32),
        np.stack(weight_records, axis=0).astype(np.float32),
    )


def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    errors = np.asarray(errors, dtype=np.float32)
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return {
            "median": float("nan"),
            "mean": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def slugify_point_id(point_id: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", point_id.strip().lower())
    slug = slug.strip("_") or "anchor"
    return slug


def anchor_distance_column(point_id: str) -> str:
    return f"pred_dist_to_{slugify_point_id(point_id)}"


def format_reference_mix(
    weights: np.ndarray, reference_df: pd.DataFrame, top_k: int = 5
) -> List[Dict[str, float]]:
    if weights.size == 0:
        return []
    order = np.argsort(weights)[::-1][:top_k]
    mix: List[Dict[str, float]] = []
    for idx in order:
        entry = reference_df.iloc[int(idx)]
        mix.append(
            {
                "point_id": entry["point_id"],
                "weight": float(weights[idx]),
                "x": float(entry["x"]),
                "y": float(entry["y"]),
            }
        )
    return mix


# ---------------------------------------------------------------------------
# CLI + orchestration
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    model_backbone_path: str
    reference_metadata_csv: str
    query_metadata_csv: str
    reference_root: Optional[str]
    query_root: Optional[str]
    feature_mode: str
    force_layout: str
    crop_len: Optional[int]
    pad_to: Optional[int]
    anchor_point_ids: List[str]
    batch_size: int
    rbf_gamma: float
    soft_top_k: Optional[int]
    report_top_k: int
    refresh_batch_norm: bool
    predictions_csv: Optional[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> InferenceConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-backbone", required=True, help="Path to the frozen PreCNN encoder .h5 file")
    parser.add_argument("--reference-metadata", required=True, help="CSV containing reference CSI window metadata")
    parser.add_argument("--query-metadata", required=True, help="CSV describing the CSI windows to localise")
    parser.add_argument("--reference-root", default=None, help="Optional root directory for reference window paths")
    parser.add_argument("--query-root", default=None, help="Optional root directory for query window paths")
    parser.add_argument(
        "--feature-mode",
        default="amp+phase+sin_cos",
        choices=sorted(SUPPORTED_FEATURE_MODES),
        help="Feature combination exported by the CSI window generator",
    )
    parser.add_argument(
        "--force-layout",
        default="TC",
        choices=["TC", "CT"],
        help="Layout of window tensors exported by MATLAB (TC = time major)",
    )
    parser.add_argument("--crop-len", type=int, default=None, help="Optional center crop before padding")
    parser.add_argument("--pad-to", type=int, default=None, help="Pad/trim windows to this length (required if crop not set)")
    parser.add_argument(
        "--anchor",
        dest="anchor_point_ids",
        action="append",
        default=None,
        help="Anchor point identifiers to report distances against (repeatable)",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding extraction")
    parser.add_argument("--rbf-gamma", type=float, default=1.0, help="RBF temperature; higher focuses on nearby anchors")
    parser.add_argument(
        "--soft-top-k",
        type=int,
        default=8,
        help="Limit the number of reference embeddings contributing to each prediction (0 = use all)",
    )
    parser.add_argument("--report-top-k", type=int, default=5, help="Number of neighbours to record in metadata output")
    parser.add_argument(
        "--no-bn-refresh",
        dest="refresh_bn",
        action="store_false",
        help="Skip batch-normalisation statistics refresh on the new session",
    )
    parser.add_argument(
        "--predictions-csv",
        default="precNN_session_predictions.csv",
        help="Where to store localisation results (omit/empty to disable)",
    )

    args = parser.parse_args(argv)

    anchor_ids = args.anchor_point_ids or ["Loc A", "Loc B"]
    soft_top_k = None if args.soft_top_k and args.soft_top_k <= 0 else args.soft_top_k
    predictions_csv = args.predictions_csv or None

    return InferenceConfig(
        model_backbone_path=args.model_backbone,
        reference_metadata_csv=args.reference_metadata,
        query_metadata_csv=args.query_metadata,
        reference_root=args.reference_root,
        query_root=args.query_root,
        feature_mode=args.feature_mode,
        force_layout=args.force_layout,
        crop_len=args.crop_len,
        pad_to=args.pad_to,
        anchor_point_ids=anchor_ids,
        batch_size=args.batch_size,
        rbf_gamma=args.rbf_gamma,
        soft_top_k=soft_top_k,
        report_top_k=args.report_top_k,
        refresh_batch_norm=args.refresh_bn,
        predictions_csv=predictions_csv,
    )


def run_localisation(config: InferenceConfig) -> pd.DataFrame:
    if config.feature_mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(
            f"Unsupported FEATURE_MODE={config.feature_mode}. Valid: {sorted(SUPPORTED_FEATURE_MODES)}"
        )

    reference_meta = load_metadata(config.reference_metadata_csv, config.reference_root)
    query_meta = load_metadata(config.query_metadata_csv, config.query_root)

    all_paths: List[str] = []
    if not reference_meta.empty:
        all_paths.extend(reference_meta["window_path"].tolist())
    if not query_meta.empty:
        all_paths.extend(query_meta["window_path"].tolist())
    all_paths = sorted({p for p in all_paths if os.path.isfile(p)})

    if not all_paths:
        raise RuntimeError(
            "No valid CSI window files discovered across reference/query metadata."
        )

    C_detected, T_min, T_max = compute_T_stats(
        all_paths, force_layout=config.force_layout, crop_len=config.crop_len
    )
    print(
        f"Detected {len(all_paths)} windows with channel count {C_detected} and time range [{T_min}, {T_max}]"
    )

    pad_length = config.pad_to if config.pad_to is not None else T_max
    print(f"Using pad length: {pad_length}")

    mu_session, sigma_session, extra_stats_session, stats_info = compute_session_norm_stats(
        all_paths, feature_mode=config.feature_mode, force_layout=config.force_layout
    )
    print("Session mu shape:", mu_session.shape, "sigma shape:", sigma_session.shape)
    print("Extra feature stats for:", stats_info.get("extra_stats_available", []))
    if stats_info.get("missing_features"):
        print("[WARN] Missing feature signals encountered:", stats_info["missing_features"])

    backbone = keras.models.load_model(config.model_backbone_path)
    backbone.trainable = False
    print("Backbone output shape:", backbone.output_shape)

    reference_tensors = np.empty((0, 0, 0), dtype=np.float32)
    if not reference_meta.empty:
        reference_tensors, reference_info = build_feature_tensor(
            reference_meta["window_path"].tolist(),
            mu_session,
            sigma_session,
            feature_mode=config.feature_mode,
            extra_stats=extra_stats_session,
            force_layout=config.force_layout,
            crop_len=config.crop_len,
            pad_to=pad_length,
        )
        print("Reference tensor shape:", reference_tensors.shape)
        print("Reference component order:", reference_info.get("component_order"))
        if reference_info.get("missing_features"):
            print("[WARN] Reference missing features:", reference_info["missing_features"])

    query_tensors = np.empty((0, 0, 0), dtype=np.float32)
    if not query_meta.empty:
        query_tensors, query_info = build_feature_tensor(
            query_meta["window_path"].tolist(),
            mu_session,
            sigma_session,
            feature_mode=config.feature_mode,
            extra_stats=extra_stats_session,
            force_layout=config.force_layout,
            crop_len=config.crop_len,
            pad_to=pad_length,
        )
        print("Query tensor shape:", query_tensors.shape)
        print("Query component order:", query_info.get("component_order"))
        if query_info.get("missing_features"):
            print("[WARN] Query missing features:", query_info["missing_features"])

    if config.refresh_batch_norm:
        calibration_batches = []
        if reference_tensors.size:
            calibration_batches.append(reference_tensors)
        if query_tensors.size:
            calibration_batches.append(query_tensors)
        if calibration_batches:
            calibration_array = np.concatenate(calibration_batches, axis=0)
            refresh_batch_norm(backbone, calibration_array, batch_size=config.batch_size)
        else:
            print("[BN] Nothing to refresh; skipping.")

    reference_embeddings = np.empty((0, backbone.output_shape[-1]), dtype=np.float32)
    if reference_tensors.size:
        reference_embeddings = extract_embeddings(
            backbone, reference_tensors, batch_size=config.batch_size
        )
        print("Reference embeddings:", reference_embeddings.shape)

    query_embeddings = np.empty((0, backbone.output_shape[-1]), dtype=np.float32)
    if query_tensors.size:
        query_embeddings = extract_embeddings(
            backbone, query_tensors, batch_size=config.batch_size
        )
        print("Query embeddings:", query_embeddings.shape)

    if reference_embeddings.size == 0:
        raise RuntimeError(
            "Reference embeddings are empty; localization requires at least one reference point."
        )

    reference_meta = reference_meta.reset_index(drop=True)
    if len(reference_meta) != len(reference_embeddings):
        raise RuntimeError(
            "Reference metadata and embeddings mismatch after extraction."
        )

    reference_database, reference_matrix, reference_coords = build_reference_database(
        reference_meta, reference_embeddings
    )
    print(reference_database[["point_id", "x", "y", "count"]])
    print("Reference matrix shape:", reference_matrix.shape)

    anchor_names: List[str] = []
    anchor_xy = np.empty((0, 2), dtype=np.float32)
    anchor_df = reference_database[
        reference_database["point_id"].isin(config.anchor_point_ids)
    ].copy()
    missing_anchors = sorted(set(config.anchor_point_ids) - set(anchor_df["point_id"]))
    if missing_anchors:
        raise RuntimeError(f"Missing anchors in reference database: {missing_anchors}")
    if not anchor_df.empty:
        anchor_df = anchor_df.set_index("point_id").loc[config.anchor_point_ids].reset_index()
        anchor_names = anchor_df["point_id"].tolist()
        anchor_xy = anchor_df[["x", "y"]].to_numpy(dtype=np.float32)
        print("Anchors for distance reporting:")
        print(anchor_df[["point_id", "x", "y", "count"]])

    if query_embeddings.size == 0:
        raise RuntimeError(
            "Query embeddings are empty; provide query metadata/windows to localize."
        )

    pred_xy, weight_matrix = soft_rbf_localization(
        query_embeddings,
        reference_matrix,
        reference_coords,
        gamma=config.rbf_gamma,
        top_k=config.soft_top_k,
    )

    query_results = query_meta.copy().reset_index(drop=True)
    query_results["pred_x"] = pred_xy[:, 0]
    query_results["pred_y"] = pred_xy[:, 1]
    true_xy = query_results[["x", "y"]].to_numpy(dtype=np.float32)
    valid_mask = np.all(np.isfinite(true_xy), axis=1)
    query_results["error_m"] = np.nan
    if valid_mask.any():
        query_results.loc[valid_mask, "error_m"] = np.linalg.norm(
            pred_xy[valid_mask] - true_xy[valid_mask], axis=1
        )

    if anchor_xy.size and anchor_names:
        anchor_dist_matrix = np.linalg.norm(
            pred_xy[:, None, :] - anchor_xy[None, :, :], axis=2
        )
        for idx, name in enumerate(anchor_names):
            col = anchor_distance_column(name)
            query_results[col] = anchor_dist_matrix[:, idx]
        query_results["closest_anchor"] = [
            anchor_names[idx] for idx in np.argmin(anchor_dist_matrix, axis=1)
        ]
        query_results["closest_anchor_distance"] = anchor_dist_matrix.min(axis=1)
    else:
        print(
            "No anchor distances computed; ensure --anchor matches reference metadata."
        )

    query_results["reference_mix"] = [
        json.dumps(
            format_reference_mix(weights, reference_database, top_k=config.report_top_k)
        )
        for weights in weight_matrix
    ]

    if np.isfinite(query_results["error_m"]).any():
        error_metrics = summarize_errors(query_results["error_m"].to_numpy())
        print("Localization error summary (meters):")
        for key, value in error_metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("No ground-truth coordinates provided; skipping error summary.")

    return query_results


def main(argv: Optional[Sequence[str]] = None) -> None:
    configure_tensorflow()
    config = parse_args(argv)
    results = run_localisation(config)
    if config.predictions_csv:
        results.to_csv(config.predictions_csv, index=False)
        print("Saved predictions to", os.path.abspath(config.predictions_csv))
    else:
        print("Predictions CSV disabled; results kept in-memory only.")


if __name__ == "__main__":
    main()
