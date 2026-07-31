"""Invariant tests for the occurrence-flag race: brake causality, counts."""

import numpy as np
from eightk_race.race import brake_series


class TestBrakeSeries:
    def test_burst_activates_forward_window_only(self):
        z = np.zeros(10)
        z[3] = 2.5
        active = brake_series(z, burst_z=2.0, brake_days=5)
        assert not active[:3].any(), "brake must never precede its burst"
        assert active[3:8].all()
        assert not active[8:].any()

    def test_subthreshold_and_nan_never_activate(self):
        z = np.array([1.99, np.nan, -3.0, 0.0])
        assert not brake_series(z, burst_z=2.0, brake_days=5).any()

    def test_overlapping_bursts_extend_the_window(self):
        z = np.zeros(12)
        z[[2, 5]] = 3.0
        active = brake_series(z, burst_z=2.0, brake_days=5)
        assert active[2:10].all()
        assert not active[0:2].any() and not active[10:].any()

    def test_burst_near_end_clips_safely(self):
        z = np.zeros(4)
        z[3] = 5.0
        active = brake_series(z, burst_z=2.0, brake_days=5)
        assert active[3] and not active[:3].any()
