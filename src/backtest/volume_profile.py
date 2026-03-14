"""Volume profile computation — POC, Value Area, HVN/LVN identification.

Computes a volume-at-price histogram from intraday bars and extracts
key levels used for confluence analysis in ORB breakout setups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VolumeProfileResult:
    """Result of a single-day volume profile computation.

    Attributes:
        poc:  Point of Control — price with highest traded volume.
        vah:  Value Area High — upper boundary of 70% volume zone.
        val_: Value Area Low  — lower boundary of 70% volume zone.
        hvn:  High Volume Nodes — price bins with volume > 1.5x average.
        lvn:  Low Volume Nodes  — price bins with volume < 0.5x average.
    """

    poc: float
    vah: float
    val_: float
    hvn: list[float] = field(default_factory=list)
    lvn: list[float] = field(default_factory=list)


def compute_volume_profile(
    high: np.ndarray[..., np.dtype[np.float64]],
    low: np.ndarray[..., np.dtype[np.float64]],
    close: np.ndarray[..., np.dtype[np.float64]],
    volume: np.ndarray[..., np.dtype[np.float64]],
    tick_size: float = 0.05,
    value_area_pct: float = 0.70,
) -> VolumeProfileResult | None:
    """Compute a volume profile from a set of OHLCV bars.

    For each bar, volume is distributed uniformly across price bins from
    the bar's low to high.  The resulting histogram is then analyzed to
    extract POC, Value Area, and HVN/LVN levels.

    Args:
        high:           Array of bar highs.
        low:            Array of bar lows.
        close:          Array of bar closes (unused, reserved for VWAP weighting).
        volume:         Array of bar volumes.
        tick_size:      Width of each price bin in dollars (default $0.05).
        value_area_pct: Fraction of total volume for Value Area (default 0.70).

    Returns:
        A :class:`VolumeProfileResult`, or ``None`` if insufficient data.
    """
    h = np.asarray(high, dtype=float)
    l_ = np.asarray(low, dtype=float)
    v = np.asarray(volume, dtype=float)
    n = len(h)

    if n < 5 or np.all(v == 0):
        return None

    # Overall price range for the day
    day_low = float(np.nanmin(l_))
    day_high = float(np.nanmax(h))

    if day_high <= day_low or tick_size <= 0:
        return None

    # Build bins
    num_bins = max(int((day_high - day_low) / tick_size) + 1, 2)
    bin_edges = np.linspace(day_low, day_high, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volume = np.zeros(num_bins, dtype=float)

    # Distribute each bar's volume across bins it spans
    for i in range(n):
        if v[i] <= 0 or np.isnan(h[i]) or np.isnan(l_[i]):
            continue
        bar_low_idx = np.searchsorted(bin_edges, l_[i], side="right") - 1
        bar_high_idx = np.searchsorted(bin_edges, h[i], side="right") - 1
        bar_low_idx = max(0, min(bar_low_idx, num_bins - 1))
        bar_high_idx = max(0, min(bar_high_idx, num_bins - 1))
        bins_spanned = bar_high_idx - bar_low_idx + 1
        vol_per_bin = v[i] / bins_spanned
        bin_volume[bar_low_idx : bar_high_idx + 1] += vol_per_bin

    total_vol = bin_volume.sum()
    if total_vol <= 0:
        return None

    # POC: bin with highest volume
    poc_idx = int(np.argmax(bin_volume))
    poc = float(bin_centers[poc_idx])

    # Value Area: expand from POC until value_area_pct of volume is captured
    va_vol = bin_volume[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx

    while va_vol / total_vol < value_area_pct:
        expand_lo = bin_volume[lo_idx - 1] if lo_idx > 0 else 0.0
        expand_hi = bin_volume[hi_idx + 1] if hi_idx < num_bins - 1 else 0.0

        if expand_lo == 0 and expand_hi == 0:
            break

        if expand_lo >= expand_hi:
            lo_idx -= 1
            va_vol += bin_volume[lo_idx]
        else:
            hi_idx += 1
            va_vol += bin_volume[hi_idx]

    val_ = float(bin_edges[lo_idx])  # lower edge of lowest VA bin
    vah = float(bin_edges[hi_idx + 1])  # upper edge of highest VA bin

    # HVN and LVN identification
    avg_vol = float(bin_volume.mean())
    hvn_threshold = avg_vol * 1.5
    lvn_threshold = avg_vol * 0.5

    hvn: list[float] = []
    lvn: list[float] = []

    for j in range(num_bins):
        if j == poc_idx:
            continue  # POC is already tracked separately
        if bin_volume[j] > hvn_threshold:
            hvn.append(float(bin_centers[j]))
        elif 0 < bin_volume[j] < lvn_threshold:
            lvn.append(float(bin_centers[j]))

    return VolumeProfileResult(
        poc=poc,
        vah=vah,
        val_=val_,
        hvn=sorted(hvn),
        lvn=sorted(lvn),
    )
