"""Core package for CSI-based sensing utilities."""

from .data import CSIDataset, NormalizationStats, compute_normalization_stats
from .model import CSISensingModel
from .utils import (
    compute_location_a_percentage,
    compute_presence_metrics,
    compute_distance_metrics,
    compute_calibration_metrics,
    PresenceMetrics,
    DistanceMetrics,
    CalibrationMetrics,
)

__all__ = [
    "CSIDataset",
    "NormalizationStats",
    "compute_normalization_stats",
    "CSISensingModel",
    "compute_location_a_percentage",
    "compute_presence_metrics",
    "compute_distance_metrics",
    "compute_calibration_metrics",
    "PresenceMetrics",
    "DistanceMetrics",
    "CalibrationMetrics",
]
