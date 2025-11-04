from __future__ import annotations

import numpy as np

from csi_sensing.utils import compute_location_a_percentage


def test_location_a_percentage_monotonic():
    distances = np.linspace(0, 500, 6)
    probs = compute_location_a_percentage(distances, sigma_cm=200.0)
    assert np.all(np.diff(probs) <= 1e-6)


def test_location_a_percentage_respects_empty_threshold():
    distances = np.array([10.0, 20.0])
    probs_empty = np.array([0.6, 0.4])
    p_a = compute_location_a_percentage(distances, sigma_cm=200.0, empty_prob=probs_empty, threshold_empty=0.5)
    assert p_a[0] == 0.0 and p_a[1] > 0.0
