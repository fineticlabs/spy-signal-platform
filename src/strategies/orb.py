"""Opening Range Breakout (ORB) strategy — 5-min window.

Filters aligned with backtest engine (engine.py) as of 2026-03-24:
- Trading window: configurable ``signal_cutoff_et`` (default 10:00 ET)
- ADX threshold: configurable ``adx_min_threshold`` (default 25)
- 2-bar consecutive candle confirmation before entry
- EMA(20) trend alignment gates direction (LONG: close > EMA, SHORT: close < EMA)
- Gap classification: ±0.3% directional gate
- VIX term structure: backwardation is a hard block (not just tag)
- ORB range minimum: 0.15% of price
"""

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

_VOL_WINDOW = 20
_VOL_MULTIPLIER = Decimal("1.5")
_ATR_MULTIPLIER = Decimal("1.5")
_RISK_MULTIPLIER = Decimal("2.0")
_VIX_MAX = Decimal("25")
_VOL_LOW_RATIO = Decimal("0.5")  # volume < 0.5 * avg → hard reject (low liquidity)
_RVOL_LOW = Decimal("0.5")  # RVOL < 0.5 → demote (-2 confidence, LOW_RVOL tag)
_RVOL_HIGH = Decimal("1.5")  # RVOL >= 1.5 → boost (+1 confidence, HIGH_RVOL tag)
_MARKET_DIRECTION_EXEMPT = {"SPY", "QQQ"}  # these ARE the market direction

# Default ORB range minimum (0.15% of price) — matches backtest _MIN_ORB_PCT
_DEFAULT_ORB_MIN_RANGE_PCT = Decimal("0.0015")
# Default gap classification threshold (±0.3%) — matches backtest _GAP_THRESHOLD_PCT
_DEFAULT_GAP_THRESHOLD_PCT = Decimal("0.3")
# Default ADX minimum — matches backtest _DAILY_ADX_MIN = 25
_DEFAULT_ADX_MIN = Decimal("25")
# Default signal cutoff — matches backtest _WINDOW1_END = 10:00 ET
_DEFAULT_CUTOFF = time(10, 0)


class ORBStrategy(Strategy):
    """5-min Opening Range Breakout strategy.

    Entry (after 2-bar confirmation):

    - LONG  when 2 consecutive closes > ORB high AND volume >= 1.5x AND EMA aligned
    - SHORT when 2 consecutive closes < ORB low  AND volume >= 1.5x AND EMA aligned

    Filters (aligned with backtest engine):

    - Excluded weekdays (default: Monday)
    - Signal cutoff (default: 10:00 ET, matching backtest 9:35-10:00 window)
    - ADX > 25 (configurable)
    - VIX < 25
    - EMA(20) trend alignment gates direction
    - Gap classification gates direction (±0.3%)
    - ORB range minimum (0.15% of price)
    - VIX backwardation hard block
    - Not in lunch chop (11:30-13:30 ET)
    - Economic calendar (FOMC/NFP/CPI/PPI days blocked)

    Stop:   ``entry +/- 1.5 * ATR(14) * kalman_mult``
    Target: ``entry +/- 2 * base_atr_risk``
    """

    def __init__(
        self,
        excluded_days: list[int] | None = None,
        signal_cutoff_et: str | None = None,
        adx_min_threshold: int | None = None,
        orb_min_range_pct: float | None = None,
        gap_threshold_pct: float | None = None,
    ) -> None:
        self._volumes: deque[int] = deque(maxlen=_VOL_WINDOW)
        self._recent_bars: deque[Bar] = deque(maxlen=5)  # for candlestick filters
        self._excluded_days: set[int] = set(excluded_days if excluded_days is not None else [0])
        self._excluded_day_logged: set[str] = set()  # track per-date logging

        # Configurable cutoff (default 10:00 ET, matching backtest)
        if signal_cutoff_et is not None:
            parts = signal_cutoff_et.split(":")
            self._cutoff = time(int(parts[0]), int(parts[1]))
        else:
            self._cutoff = _DEFAULT_CUTOFF

        # Configurable ADX threshold (default 25, matching backtest)
        self._adx_min = (
            Decimal(str(adx_min_threshold)) if adx_min_threshold is not None else _DEFAULT_ADX_MIN
        )

        # ORB range minimum
        self._orb_min_range_pct = (
            Decimal(str(orb_min_range_pct))
            if orb_min_range_pct is not None
            else _DEFAULT_ORB_MIN_RANGE_PCT
        )

        # Gap classification threshold
        self._gap_threshold_pct = (
            Decimal(str(gap_threshold_pct))
            if gap_threshold_pct is not None
            else _DEFAULT_GAP_THRESHOLD_PCT
        )

        # 2-bar confirmation state: symbol → (direction, bar_timestamp)
        self._pending_breakouts: dict[str, tuple[Direction, str]] = {}

        # Gap classification cache: date_str → gap_pct (computed once per day per symbol)
        self._gap_cache: dict[str, Decimal | None] = {}

        # ORB range filter cache: date_str:symbol → blocked (computed once per day)
        self._orb_range_blocked: dict[str, bool] = {}

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
        """Return a Signal if ORB entry conditions are met, else ``None``."""
        # --- Excluded weekday filter (Monday by default) ---
        bar_date = bar.timestamp.astimezone(_ET).date()
        if bar_date.weekday() in self._excluded_days:
            date_key = str(bar_date)
            if date_key not in self._excluded_day_logged:
                day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
                day_name = day_names.get(bar_date.weekday(), str(bar_date.weekday()))
                logger.info(
                    "orb_filter_excluded_day",
                    date=date_key,
                    day=day_name,
                    symbol=bar.symbol,
                )
                self._excluded_day_logged.add(date_key)
            return None

        # Capture avg BEFORE this bar contaminates the window, then record it.
        avg_vol = self._avg_volume()
        self._volumes.append(bar.volume)
        self._recent_bars.append(bar)

        bar_time = bar.timestamp.astimezone(_ET).time()

        # --- Time filters (cutoff aligned with backtest: default 10:00 ET) ---
        if bar_time >= self._cutoff:
            logger.debug("orb_filter_cutoff", bar_time=str(bar_time), cutoff=str(self._cutoff))
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

        # --- ORB range minimum filter (matches backtest _MIN_ORB_PCT) ---
        orb_range_key = f"{bar_date}:{bar.symbol}"
        if orb_range_key not in self._orb_range_blocked:
            orb_range = orb_high - orb_low
            range_pct = orb_range / orb_low if orb_low > 0 else Decimal("0")
            blocked = range_pct < self._orb_min_range_pct
            self._orb_range_blocked[orb_range_key] = blocked
            if blocked:
                logger.info(
                    "orb_filter_narrow_range",
                    symbol=bar.symbol,
                    orb_high=str(orb_high),
                    orb_low=str(orb_low),
                    range_pct=str(range_pct),
                    min_pct=str(self._orb_min_range_pct),
                )
        if self._orb_range_blocked.get(orb_range_key, False):
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
        if adx <= self._adx_min:
            logger.debug("orb_filter_low_adx", adx=str(adx), threshold=str(self._adx_min))
            return None

        # --- VIX term structure: backwardation is a HARD BLOCK (matches backtest) ---
        if vix_term_ratio is not None:
            from src.filters.vix_term_structure import BACKWARDATION_THRESHOLD

            if float(vix_term_ratio) > BACKWARDATION_THRESHOLD:
                logger.info(
                    "orb_filter_vix_backwardation",
                    symbol=bar.symbol,
                    vix_term_ratio=str(vix_term_ratio),
                )
                return None

        # --- Low volume rejection (hard gate) ---
        bar_vol = Decimal(str(bar.volume))
        if avg_vol is not None and bar_vol < avg_vol * _VOL_LOW_RATIO:
            logger.info(
                "orb_filter_low_volume",
                symbol=bar.symbol,
                volume=bar.volume,
                avg=str(avg_vol),
                ratio=str(bar_vol / avg_vol if avg_vol else 0),
            )
            return None

        # --- Breakout volume confirmation ---
        if avg_vol is None or bar_vol < avg_vol * _VOL_MULTIPLIER:
            logger.debug("orb_filter_volume", volume=bar.volume, avg=str(avg_vol))
            return None

        # --- ATR required for stop sizing ---
        atr = indicators.atr
        if atr is None:
            logger.debug("orb_filter_no_atr")
            return None

        close = bar.close

        # --- EMA trend alignment (matches backtest 15m EMA(20) gate) ---
        ema20 = indicators.ema20
        if ema20 is not None:
            if close > orb_high and close <= ema20:
                logger.info(
                    "orb_filter_ema_trend",
                    symbol=bar.symbol,
                    direction="LONG",
                    close=str(close),
                    ema20=str(ema20),
                )
                return None
            if close < orb_low and close >= ema20:
                logger.info(
                    "orb_filter_ema_trend",
                    symbol=bar.symbol,
                    direction="SHORT",
                    close=str(close),
                    ema20=str(ema20),
                )
                return None

        # --- Gap classification (matches backtest ±0.3% directional gate) ---
        gap_pct = self._get_gap_pct(bar, levels)
        if gap_pct is not None:
            if close > orb_high and gap_pct < -self._gap_threshold_pct:
                # Gap down but trying to go LONG — blocked
                logger.info(
                    "orb_filter_gap_direction",
                    symbol=bar.symbol,
                    direction="LONG",
                    gap_pct=str(gap_pct),
                )
                return None
            if close < orb_low and gap_pct > self._gap_threshold_pct:
                # Gap up but trying to go SHORT — blocked
                logger.info(
                    "orb_filter_gap_direction",
                    symbol=bar.symbol,
                    direction="SHORT",
                    gap_pct=str(gap_pct),
                )
                return None

        # --- Determine breakout direction ---
        if close > orb_high:
            current_direction = Direction.LONG
        elif close < orb_low:
            current_direction = Direction.SHORT
        else:
            # Price inside ORB range — clear any pending breakout
            if bar.symbol in self._pending_breakouts:
                logger.info(
                    "orb_failed_confirmation",
                    symbol=bar.symbol,
                    reason="price_back_inside_orb",
                )
                del self._pending_breakouts[bar.symbol]
            return None

        # --- 2-bar consecutive candle confirmation (matches backtest) ---
        bar_ts_key = bar.timestamp.isoformat()
        pending = self._pending_breakouts.get(bar.symbol)

        if pending is None or pending[0] != current_direction:
            # First bar breaking ORB in this direction — set pending
            self._pending_breakouts[bar.symbol] = (current_direction, bar_ts_key)
            logger.info(
                "orb_pending_confirmation",
                symbol=bar.symbol,
                direction=str(current_direction),
                close=str(close),
            )
            return None

        # Second consecutive bar outside ORB in same direction — confirmed!
        del self._pending_breakouts[bar.symbol]
        direction = current_direction
        logger.info(
            "orb_confirmed_breakout",
            symbol=bar.symbol,
            direction=str(direction),
            close=str(close),
        )

        # --- Build signal ---
        km = kalman_stop_mult if kalman_stop_mult is not None else Decimal("1.0")
        adaptive_atr_stop = _ATR_MULTIPLIER * atr * km
        base_atr_risk = _ATR_MULTIPLIER * atr  # original risk for target (decoupled)

        if direction == Direction.LONG:
            entry = close
            stop = entry - adaptive_atr_stop
            risk = entry - stop
            target = entry + _RISK_MULTIPLIER * base_atr_risk
            reason = f"Close {close} broke above ORB high {orb_high} (2-bar confirmed) with volume {bar.volume:,}"
        else:
            entry = close
            stop = entry + adaptive_atr_stop
            risk = stop - entry
            target = entry - _RISK_MULTIPLIER * base_atr_risk
            reason = f"Close {close} broke below ORB low {orb_low} (2-bar confirmed) with volume {bar.volume:,}"

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

        # Gap classification tag (informational)
        if gap_pct is not None:
            if gap_pct > self._gap_threshold_pct:
                tags.append("GAP_UP")
            elif gap_pct < -self._gap_threshold_pct:
                tags.append("GAP_DOWN")
            else:
                tags.append("GAP_FLAT")

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

        # VIX term structure contango tag (backwardation already hard-blocked above)
        if vix_term_ratio is not None:
            from src.filters.vix_term_structure import CONTANGO_THRESHOLD

            ratio_f = float(vix_term_ratio)
            if ratio_f < CONTANGO_THRESHOLD:
                confidence += 1
                tags.append("CONTANGO")

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

        confidence = max(1, min(confidence, 5))

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
            symbol=bar.symbol,
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

    def _get_gap_pct(self, bar: Bar, levels: LevelSnapshot) -> Decimal | None:
        """Compute today's gap percentage vs previous day close.

        Returns gap as a percentage (e.g., 0.5 = +0.5%).
        Cached per date:symbol so it's only computed once.
        """
        bar_date = bar.timestamp.astimezone(_ET).date()
        cache_key = f"{bar_date}:{bar.symbol}"
        if cache_key in self._gap_cache:
            return self._gap_cache[cache_key]

        prev_close = levels.prev_day_close
        if prev_close is None or prev_close <= 0:
            self._gap_cache[cache_key] = None
            return None

        # Approximate session open from ORB midpoint (the true session open is
        # the first bar's open, which we don't store separately).
        orb_high = levels.orb_high
        orb_low = levels.orb_low
        if orb_high is not None and orb_low is not None:
            # Approximate session open as ORB midpoint (close enough for gap calc)
            # The true session open is the first bar's open, which we don't store
            # separately. The ORB midpoint is a reasonable proxy.
            approx_open = (orb_high + orb_low) / 2
        else:
            approx_open = bar.open

        gap_pct = (approx_open - prev_close) / prev_close * 100
        self._gap_cache[cache_key] = gap_pct

        logger.info(
            "orb_gap_classified",
            symbol=bar.symbol,
            gap_pct=str(gap_pct),
            prev_close=str(prev_close),
            approx_open=str(approx_open),
            direction="UP"
            if gap_pct > self._gap_threshold_pct
            else ("DOWN" if gap_pct < -self._gap_threshold_pct else "FLAT"),
        )
        return gap_pct
