"""Opening Range Breakout (ORB) strategy — 5-min window."""

from __future__ import annotations

from collections import deque
from datetime import time
from decimal import Decimal
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

from src.filters.earnings_calendar import is_earnings_blackout
from src.filters.economic_calendar import is_high_impact_day
from src.models import Bar, Direction, Signal
from src.strategies.base import Strategy
from src.strategies.candlestick_filters import (
    has_inside_bar_compression,
    is_engulfing,
)

if TYPE_CHECKING:
    from src.models import IndicatorSnapshot, LevelSnapshot
    from src.strategies.regime import RegimeDetector

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_LUNCH_START = time(11, 30)
_LUNCH_END = time(13, 30)
_CUTOFF = time(15, 45)

_VOL_WINDOW = 20
_VOL_MULTIPLIER = Decimal("1.5")
_ATR_MULTIPLIER = Decimal("1.5")
_RISK_MULTIPLIER = Decimal("2.0")
_VIX_MAX = Decimal("25")
_ADX_MIN = Decimal("20")
_RVOL_LOW = Decimal("0.5")  # RVOL < 0.5 → demote (-2 confidence, LOW_RVOL tag)
_RVOL_HIGH = Decimal("1.5")  # RVOL >= 1.5 → boost (+1 confidence, HIGH_RVOL tag)
_MARKET_DIRECTION_EXEMPT = {"SPY", "QQQ"}  # these ARE the market direction


class ORBStrategy(Strategy):
    """5-min Opening Range Breakout strategy.

    Entry:

    - LONG  when ``close > ORB high`` AND ``volume >= 1.5x 20-bar average``
    - SHORT when ``close < ORB low``  AND ``volume >= 1.5x 20-bar average``

    Filters:

    - ORB window complete (>= 9:35 ET)
    - VIX < 25
    - ADX > 20
    - Not in lunch chop (11:30-13:30 ET)
    - Before forced-flat cutoff (15:45 ET)

    Stop:   ``entry +/- 1.5 * ATR(14)``
    Target: ``entry +/- 2 * risk_distance``
    """

    def __init__(self) -> None:
        self._volumes: deque[int] = deque(maxlen=_VOL_WINDOW)
        self._recent_bars: deque[Bar] = deque(maxlen=5)  # for candlestick filters

    # ── Strategy interface ─────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ORB-5min"

    def required_indicators(self) -> list[str]:
        return ["atr"]

    def evaluate(
        self,
        bar: Bar,
        indicators: IndicatorSnapshot,
        levels: LevelSnapshot,
        regime: RegimeDetector,
        spy_vwap: Decimal | None = None,
        spy_price: Decimal | None = None,
        vp_poc: Decimal | None = None,
        vp_vah: Decimal | None = None,
        vp_val: Decimal | None = None,
        vp_hvn: Decimal | None = None,
        vp_lvn: Decimal | None = None,
        vix_term_ratio: Decimal | None = None,
        hmm_regime: str | None = None,
        kalman_stop_mult: Decimal | None = None,
    ) -> Signal | None:
        """Return a Signal if ORB entry conditions are met, else ``None``.

        Volume history is updated on every call regardless of filtering so that
        the rolling average stays current even while conditions are not met.

        Args:
            bar:        Current completed bar.
            indicators: Indicator snapshot at signal time.
            levels:     Level snapshot (ORB, VWAP, etc.).
            regime:     Regime detector with VIX/ADX state.
            spy_vwap:   SPY's current intraday VWAP (for market direction
                        confirmation on non-SPY/QQQ tickers).
            spy_price:  SPY's current price (last close).
        """
        # Capture avg BEFORE this bar contaminates the window, then record it.
        avg_vol = self._avg_volume()
        self._volumes.append(bar.volume)
        self._recent_bars.append(bar)

        bar_time = bar.timestamp.astimezone(_ET).time()

        # --- Time filters ---
        if bar_time >= _CUTOFF:
            logger.debug("orb_filter_cutoff", bar_time=str(bar_time))
            return None
        if _LUNCH_START <= bar_time < _LUNCH_END:
            logger.debug("orb_filter_lunch_chop", bar_time=str(bar_time))
            return None

        # --- Economic calendar filter: skip FOMC/NFP/CPI/PPI days ---
        bar_date = bar.timestamp.astimezone(_ET).date()
        if is_high_impact_day(bar_date):
            logger.debug("orb_filter_econ_event", date=str(bar_date))
            return None

        # --- Earnings proximity: informational tag, no blocking ---
        _is_earnings = is_earnings_blackout(bar.symbol, bar_date)

        # --- ORB must be complete ---
        if not levels.orb_complete:
            logger.debug("orb_filter_incomplete")
            return None

        orb_high = levels.orb_high
        orb_low = levels.orb_low
        if orb_high is None or orb_low is None:
            return None

        # --- Regime filters ---
        vix = regime.vix_level
        adx = regime.adx_value
        if vix is None or adx is None:
            logger.debug("orb_filter_no_regime_data")
            return None
        if vix >= _VIX_MAX:
            logger.debug("orb_filter_high_vix", vix=str(vix))
            return None
        if adx <= _ADX_MIN:
            logger.debug("orb_filter_low_adx", adx=str(adx))
            return None

        # --- Volume filter ---
        if avg_vol is None or Decimal(str(bar.volume)) < avg_vol * _VOL_MULTIPLIER:
            logger.debug("orb_filter_volume", volume=bar.volume, avg=str(avg_vol))
            return None

        # --- ATR required for stop sizing ---
        atr = indicators.atr
        if atr is None:
            logger.debug("orb_filter_no_atr")
            return None

        close = bar.close

        # --- Kalman-adaptive stop sizing ---
        km = kalman_stop_mult if kalman_stop_mult is not None else Decimal("1.0")
        adaptive_atr_stop = _ATR_MULTIPLIER * atr * km
        base_atr_risk = _ATR_MULTIPLIER * atr  # original risk for target (decoupled)

        # --- Determine direction ---
        if close > orb_high:
            direction = Direction.LONG
            entry = close
            stop = entry - adaptive_atr_stop
            risk = entry - stop
            target = entry + _RISK_MULTIPLIER * base_atr_risk
            reason = f"Close {close} broke above ORB high {orb_high} " f"with volume {bar.volume:,}"
        elif close < orb_low:
            direction = Direction.SHORT
            entry = close
            stop = entry + adaptive_atr_stop
            risk = stop - entry
            target = entry - _RISK_MULTIPLIER * base_atr_risk
            reason = f"Close {close} broke below ORB low {orb_low} " f"with volume {bar.volume:,}"
        else:
            return None  # price inside ORB range — no breakout

        # --- Market direction confirmation: informational tag, no blocking ---
        _spy_aligned: bool | None = None
        if (
            bar.symbol not in _MARKET_DIRECTION_EXEMPT
            and spy_vwap is not None
            and spy_price is not None
        ):
            if direction == Direction.LONG:
                _spy_aligned = spy_price > spy_vwap
            else:
                _spy_aligned = spy_price < spy_vwap

        if risk <= 0:
            return None

        # ── Candlestick quality boosters ─────────────────────────────────
        dir_str = str(direction)
        confidence = 3
        tags: list[str] = []

        # SPY VWAP alignment tag (informational, no blocking)
        if _spy_aligned is True:
            confidence += 1
            tags.append("SPY_ALIGNED")
        elif _spy_aligned is False:
            confidence -= 2
            tags.append("SPY_CONFLICT")

        # Earnings proximity tag (informational, no blocking)
        if _is_earnings:
            tags.append("EARNINGS")

        # RVOL confidence adjustment (informational, no blocking)
        rvol = levels.rvol
        if rvol is not None:
            if rvol < _RVOL_LOW:
                confidence -= 2
                tags.append("LOW_RVOL")
            elif rvol >= _RVOL_HIGH:
                confidence += 1
                tags.append("HIGH_RVOL")

        # Volume profile confidence adjustment (informational, no blocking)
        if vp_poc is not None:
            # LVN target: target beyond Value Area → low-volume zone, less resistance
            if vp_vah is not None and vp_val is not None:
                if target > vp_vah or target < vp_val:
                    confidence += 1
                    tags.append("VP_LVN_TARGET")
                # HVN target: target inside Value Area → high-volume resistance
                elif vp_val < target < vp_vah:
                    confidence -= 1
                    tags.append("VP_HVN_TARGET")
            # POC cross: entry-to-target path crosses POC → potential barrier
            if (close < vp_poc < target) or (target < vp_poc < close):
                confidence -= 1
                tags.append("VP_POC_CROSS")

        # VIX term structure confidence adjustment (informational, no blocking)
        if vix_term_ratio is not None:
            from src.filters.vix_term_structure import (
                BACKWARDATION_THRESHOLD,
                CONTANGO_THRESHOLD,
            )

            ratio_f = float(vix_term_ratio)
            if ratio_f < CONTANGO_THRESHOLD:
                confidence += 1
                tags.append("CONTANGO")
            elif ratio_f > BACKWARDATION_THRESHOLD:
                confidence -= 2
                tags.append("BACKWARDATION")

        # HMM regime confidence adjustment (informational, no blocking)
        # Backtest showed VOLATILE is best for ORB (+1), CALM is weakest (-1)
        if hmm_regime is not None:
            if hmm_regime == "VOLATILE":
                confidence += 1
                tags.append("HMM_VOLATILE")
            elif hmm_regime == "CALM":
                confidence -= 1
                tags.append("HMM_CALM")
            elif hmm_regime == "NORMAL":
                tags.append("HMM_NORMAL")

        # Engulfing bar: boost confidence if breakout candle engulfs prior
        if len(self._recent_bars) >= 2:
            prev = self._recent_bars[-2]
            if is_engulfing(
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(prev.open),
                float(prev.close),
                dir_str,
            ):
                confidence += 1
                tags.append("ENGULFING")

        # Inside bar compression: boost confidence if compression before breakout
        if len(self._recent_bars) >= 5:
            highs = [float(b.high) for b in list(self._recent_bars)[:-1]]
            lows = [float(b.low) for b in list(self._recent_bars)[:-1]]
            if has_inside_bar_compression(highs, lows, lookback=3):
                confidence += 1
                tags.append("COMPRESSED")

        confidence = min(confidence, 5)

        rr = _RISK_MULTIPLIER  # always 2.0 by construction

        signal = Signal(
            direction=direction,
            strategy_name=self.name,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            risk_reward_ratio=rr,
            confidence_score=confidence,
            reason=reason,
            timeframe=bar.timeframe,
            regime=regime.current_regime,
            vix=vix,
            adx=adx,
            indicators_snapshot=indicators,
            levels_snapshot=levels,
            timestamp=bar.timestamp,
            tags=tags,
        )

        logger.info(
            "orb_signal",
            direction=dir_str,
            entry=str(entry),
            stop=str(stop),
            target=str(target),
            rr=str(rr),
            tags=tags,
        )
        return signal

    # ── helpers ────────────────────────────────────────────────────────────────

    def _avg_volume(self) -> Decimal | None:
        """Rolling average volume, or ``None`` when the window is empty."""
        if not self._volumes:
            return None
        return Decimal(str(sum(self._volumes) / len(self._volumes)))
