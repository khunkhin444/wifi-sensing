#!/usr/bin/env python3
"""Utility for building PreCNN localisation metadata CSV files.

The CSI_PreCNN_Inference notebook and CLI expect metadata CSV files with four
columns: ``window_path``, ``point_id``, ``x``, and ``y``.  This helper creates
those CSVs either from the ``split_list.csv`` emitted by ``Allocation.py`` or by
enumerating CSI window files directly from a directory structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("window_path", "point_id", "x", "y")
SPLIT_DEFAULT_COLUMN_NAMES = {
    "split": ("Split", "split"),
    "label": ("Label", "Class", "class", "label"),
    "path": ("File Path", "filepath", "file_path", "window_path", "path"),
}
COORD_DEFAULT_COLUMN_NAMES = {
    "point_id": ("point_id", "PointID", "Point Id", "Label", "label"),
    "x": ("x", "X", "coord_x", "Coord X", "coordX"),
    "y": ("y", "Y", "coord_y", "Coord Y", "coordY"),
}


class MetadataBuildError(RuntimeError):
    """Raised when metadata generation fails due to invalid input."""


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first matching column name from ``candidates``.

    Matches are case-insensitive and ignore whitespace/underscore differences.
    """

    normalised = {
        "".join(filter(str.isalnum, col.lower())): col for col in df.columns
    }
    for candidate in candidates:
        key = "".join(filter(str.isalnum, candidate.lower()))
        if key in normalised:
            return normalised[key]
    raise MetadataBuildError(
        f"Could not find any of the columns {candidates!r} in {list(df.columns)!r}."
    )


def _load_split_list(
    path: Path, splits: Optional[Sequence[str]] = None, drop_missing: bool = False
) -> pd.DataFrame:
    df = pd.read_csv(path)
    split_col = _resolve_column(df, SPLIT_DEFAULT_COLUMN_NAMES["split"])
    label_col = _resolve_column(df, SPLIT_DEFAULT_COLUMN_NAMES["label"])
    path_col = _resolve_column(df, SPLIT_DEFAULT_COLUMN_NAMES["path"])

    if splits:
        splits_lower = {s.lower() for s in splits}
        df = df[df[split_col].astype(str).str.lower().isin(splits_lower)]
        if df.empty:
            raise MetadataBuildError(
                "No rows matched the requested splits. Please check the --split argument."
            )

    if drop_missing:
        df = df[df[path_col].notna()]

    df = df[[path_col, label_col]].copy()
    df.columns = ["window_path", "point_id"]
    df["window_path"] = df["window_path"].astype(str)
    df["point_id"] = df["point_id"].astype(str)
    return df


def _enumerate_windows(
    root: Path,
    pattern: str = "X_window_*.npy",
    recursive: bool = False,
    point_source: str = "parent",
    keep_empty: bool = False,
) -> pd.DataFrame:
    if not root.is_dir():
        raise MetadataBuildError(f"Window root does not exist or is not a directory: {root}")

    iterator: Iterable[Path]
    if recursive:
        iterator = root.rglob(pattern)
    else:
        iterator = root.glob(pattern)

    records = []
    for path in sorted(iterator):
        if not path.is_file():
            continue
        if point_source == "parent":
            point_id = path.parent.name
        elif point_source == "stem":
            point_id = path.stem
        else:
            raise MetadataBuildError("point_source must be 'parent' or 'stem'")
        if not point_id and not keep_empty:
            raise MetadataBuildError(
                f"Could not infer point_id for {path}; specify --point-source or reorganise directories."
            )
        records.append({"window_path": str(path), "point_id": point_id})

    if not records:
        raise MetadataBuildError(
            "No CSI windows matched the provided directory/pattern combination."
        )
    return pd.DataFrame.from_records(records)


def _load_coordinate_mapping(path: Path) -> Dict[str, Tuple[float, float]]:
    if not path.is_file():
        raise MetadataBuildError(f"Coordinate mapping file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, Mapping):
            items = payload.items()
        elif isinstance(payload, Sequence):
            items = []
            for entry in payload:
                if not isinstance(entry, Mapping):
                    raise MetadataBuildError(
                        "JSON coordinate entries must be objects with point_id/x/y keys."
                    )
                items.append((entry.get("point_id"), entry))
        else:
            raise MetadataBuildError("Unsupported JSON structure for coordinates.")
        mapping: Dict[str, Tuple[float, float]] = {}
        for key, value in items:
            if key is None:
                if isinstance(value, Mapping) and "point_id" in value:
                    key = value["point_id"]
                else:
                    raise MetadataBuildError(
                        "Coordinate JSON objects require a 'point_id' field."
                    )
            if not isinstance(value, Mapping):
                raise MetadataBuildError(
                    "Coordinate JSON dictionary values must be objects with 'x' and 'y'."
                )
            try:
                x = float(value["x"])
                y = float(value["y"])
            except (TypeError, ValueError, KeyError) as exc:
                raise MetadataBuildError(
                    f"Invalid coordinate entry for {key!r}: {value!r}"
                ) from exc
            mapping[str(key)] = (x, y)
        return mapping

    df = pd.read_csv(path)
    point_col = _resolve_column(df, COORD_DEFAULT_COLUMN_NAMES["point_id"])
    x_col = _resolve_column(df, COORD_DEFAULT_COLUMN_NAMES["x"])
    y_col = _resolve_column(df, COORD_DEFAULT_COLUMN_NAMES["y"])

    mapping: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        key = str(row[point_col])
        try:
            x = float(row[x_col])
            y = float(row[y_col])
        except (TypeError, ValueError) as exc:
            raise MetadataBuildError(
                f"Invalid coordinate values for {key!r}: x={row[x_col]!r} y={row[y_col]!r}"
            ) from exc
        mapping[key] = (x, y)
    if not mapping:
        raise MetadataBuildError(
            "Coordinate mapping file did not contain any valid rows."
        )
    return mapping


def _apply_coordinate_mapping(
    df: pd.DataFrame,
    mapping: Mapping[str, Tuple[float, float]],
    missing: str = "keep",
) -> pd.DataFrame:
    if missing not in {"keep", "drop", "error"}:
        raise MetadataBuildError("missing strategy must be 'keep', 'drop', or 'error'")

    coord_x = []
    coord_y = []
    missing_ids = []
    for point_id in df["point_id"].astype(str):
        if point_id in mapping:
            x, y = mapping[point_id]
            coord_x.append(float(x))
            coord_y.append(float(y))
        else:
            coord_x.append(np.nan)
            coord_y.append(np.nan)
            missing_ids.append(point_id)

    df = df.copy()
    df["x"] = coord_x
    df["y"] = coord_y

    if missing_ids:
        unique_missing = sorted(set(missing_ids))
        if missing == "error":
            raise MetadataBuildError(
                "Missing coordinates for point_id(s): " + ", ".join(unique_missing)
            )
        if missing == "drop":
            df = df[~df["point_id"].isin(unique_missing)]
            if df.empty:
                raise MetadataBuildError(
                    "All rows were dropped because no coordinates were available."
                )
        else:
            print(
                "[WARN] Missing coordinates for point_id(s): "
                + ", ".join(unique_missing)
            )
    else:
        print("Loaded coordinate mapping for all point_ids.")
    return df


def _ensure_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "x" not in df.columns:
        df["x"] = np.nan
    if "y" not in df.columns:
        df["y"] = np.nan
    return df


def _make_paths_relative(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    root = root.resolve()
    rows = []
    for value in df["window_path"].astype(str):
        path = Path(value)
        try:
            rel = path.resolve().relative_to(root)
        except Exception as exc:  # noqa: BLE001
            raise MetadataBuildError(
                f"Cannot make path {path} relative to {root}: {exc}"
            ) from exc
        rows.append(rel.as_posix())
    df = df.copy()
    df["window_path"] = rows
    return df


def build_metadata(args: argparse.Namespace) -> pd.DataFrame:
    if args.split_list:
        df = _load_split_list(
            Path(args.split_list), splits=args.split, drop_missing=args.drop_missing
        )
    elif args.windows_root:
        df = _enumerate_windows(
            Path(args.windows_root),
            pattern=args.pattern,
            recursive=args.recursive,
            point_source=args.point_source,
            keep_empty=args.keep_empty_point_ids,
        )
    else:
        raise MetadataBuildError("You must provide either --split-list or --windows-root.")

    if args.relative_to:
        df = _make_paths_relative(df, Path(args.relative_to))

    if args.coords:
        mapping = _load_coordinate_mapping(Path(args.coords))
        df = _apply_coordinate_mapping(df, mapping, missing=args.missing_coords)
    else:
        df = _ensure_coordinate_columns(df)

    if args.sort_by_point:
        df = df.sort_values(["point_id", "window_path"]).reset_index(drop=True)
    else:
        df = df.sort_values("window_path").reset_index(drop=True)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise MetadataBuildError(
            f"Generated metadata is missing required columns: {missing_cols}"
        )
    return df


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metadata CSV files compatible with CSI PreCNN inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--split-list",
        type=str,
        help="Path to split_list.csv produced by Allocation.py",
    )
    source.add_argument(
        "--windows-root",
        type=str,
        help="Directory to enumerate CSI windows from (expects subfolders per point).",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination CSV path for generated metadata.",
    )
    parser.add_argument(
        "--split",
        action="append",
        help="Restrict split_list.csv rows to specific split names (can repeat).",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows with missing file paths when reading split_list.csv.",
    )
    parser.add_argument(
        "--pattern",
        default="X_window_*.npy",
        help="Glob pattern when enumerating windows from a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for windows under --windows-root.",
    )
    parser.add_argument(
        "--point-source",
        choices=["parent", "stem"],
        default="parent",
        help="How to derive point_id when enumerating windows from a directory.",
    )
    parser.add_argument(
        "--keep-empty-point-ids",
        action="store_true",
        help="Allow empty point_id strings when enumerating windows.",
    )
    parser.add_argument(
        "--coords",
        type=str,
        help="Optional CSV or JSON file that maps point_id to x/y coordinates.",
    )
    parser.add_argument(
        "--missing-coords",
        choices=["keep", "drop", "error"],
        default="keep",
        help="How to handle rows missing coordinates after applying --coords.",
    )
    parser.add_argument(
        "--relative-to",
        type=str,
        help="Make window_path values relative to this directory (useful with notebook roots).",
    )
    parser.add_argument(
        "--sort-by-point",
        action="store_true",
        help="Sort output first by point_id, then window_path (default sorts by path only).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        df = build_metadata(args)
    except MetadataBuildError as exc:
        parser = argparse.ArgumentParser(add_help=False)
        parser.exit(status=2, message=f"Error: {exc}\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Metadata CSV written to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
