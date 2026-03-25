"""Tests for StreamingKalman stop multiplier."""

from __future__ import annotations

import numpy as np

from src.levels.kalman_levels import StreamingKalman, _run_kalman_for_day


class TestStreamingKalman:
    def test_default_multiplier_is_one(self) -> None:
        km = StreamingKalman()
        assert km.multiplier == 1.0
        assert not km.ready

    def test_reset_initializes_filter(self) -> None:
        km = StreamingKalman()
        km.reset([100.0, 100.5, 99.8, 100.2, 100.1], atr_value=0.5)
        assert km.ready
        assert km.multiplier == 1.0  # no updates yet

    def test_reset_with_bad_atr_stays_unready(self) -> None:
        km = StreamingKalman()
        km.reset([100.0] * 5, atr_value=0.0)
        assert not km.ready

    def test_update_changes_multiplier(self) -> None:
        km = StreamingKalman()
        km.reset([100.0, 100.5, 99.8, 100.2, 100.1], atr_value=0.5)
        # Feed some post-ORB bars
        for price in [100.3, 100.6, 100.9, 101.2, 101.5]:
            km.update(price)
        # After updates, multiplier should deviate from 1.0
        assert km.multiplier != 1.0

    def test_multiplier_clamped(self) -> None:
        km = StreamingKalman()
        km.reset([100.0, 100.5, 99.8, 100.2, 100.1], atr_value=0.5)
        # Feed extreme price moves
        for price in [110.0, 120.0, 130.0, 140.0, 150.0]:
            km.update(price)
        assert km.multiplier <= 1.5  # clamped at max
        assert km.multiplier >= 0.9  # clamped at min

    def test_matches_batch_for_calm_day(self) -> None:
        """Streaming and batch produce same result for the same input."""
        orb_closes = [100.0, 100.1, 99.9, 100.2, 100.0]
        post_orb = [100.1, 100.15, 100.2, 100.18, 100.25, 100.3, 100.28]
        all_closes = np.array(orb_closes + post_orb, dtype=float)
        atr = 0.3

        # Batch
        batch_mult = _run_kalman_for_day(all_closes, atr, orb_bars=5)

        # Streaming
        km = StreamingKalman(orb_bars=5)
        km.reset(orb_closes, atr)
        streaming_mults = []
        for p in post_orb:
            km.update(p)
            streaming_mults.append(km.multiplier)

        # Compare last values (they should match closely)
        assert abs(streaming_mults[-1] - batch_mult[-1]) < 0.01

    def test_update_before_ready_returns_default(self) -> None:
        km = StreamingKalman()
        result = km.update(100.0)
        assert result == 1.0
