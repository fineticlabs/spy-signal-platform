"""Failed Breakout reversal strategy.

``FailedBreakoutStrategy`` monitors for price that pokes above the ORB high
(or below the ORB low) by 0.5-1.5 ATR — the hallmark of a fake breakout that
traps momentum traders — then reverses back inside the opening range within 5
bars with a confirming volume spike.

Entry logic
-----------
1. **Setup bar**: close in ``[orb_level ± 0.5*ATR, orb_level ± 1.5*ATR]``.
   This identifies a poke outside the ORB that is too small to be a real breakout
   but large enough to trap breakout buyers/sellers.
2. **Reversal confirmation** (within the next 5 bars): price closes back *inside*
   the ORB range **and** volume >= 1.5x 20-bar average.  High volume on the
   reversal candle signals trapped traders exiting en masse.

Trade details
-------------
- Direction: OPPOSITE of the failed breakout direction.
- Stop:      Extreme of the failed breakout (highest close for a failed long;
             lowest close for a failed short).
- Target:    Opposite ORB boundary (ORB low for a failed long; ORB high for a
             failed short).
- Minimum R:R: 1.5:1 or the signal is skipped.

Filters
-------
- ORB must be complete (≥ 9:35 ET).
- VIX < 25, ADX > 20 (same regime gates as ORB).
- Before 15:45 ET forced-flat cutoff.
"""

from __future__ import annotations

from collections import deque
from datetime import date, time
from decimal import Decimal
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

from src.models import Direction, Signal
from src.strategies.base import Strategy

if TYPE_CHECKING:
    from src.models import Bar, IndicatorSnapshot, LevelSnapshot
    from src.strategies.regime import RegimeDetector

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_CUTOFF = time(15, 45)

_VOL_WINDOW = 20
_VOL_MULTIPLIER = Decimal("1.5")
_ATR_MIN_EXT = Decimal("0.5")  # min ATR extension past ORB for a fake breakout
_ATR_MAX_EXT = Decimal("1.5")  # max ATR extension — beyond this = real breakout
_MIN_RR = Decimal("1.5")  # minimum required reward-to-risk ratio
_REVERSAL_WINDOW = 5  # max bars to wait for reversal after setup
_VIX_MAX = Decimal("25")
_ADX_MIN = Decimal("20")


class FailedBreakoutStrategy(Strategy):
    """Failed Breakout reversal strategy.

    Detects ORB fake-outs: price pokes 0.5-1.5 ATR past an ORB level,
    then reverses back inside the range with a confirming volume spike.
    Trades the trapped-trader reversal in the *opposite* direction.

    State is maintained across ``evaluate`` calls; call with bars in
    chronological order.
    """

    def __init__(self) -> None:
        self._volumes: deque[int] = deque(maxlen=_VOL_WINDOW)

        # Failed-breakout watch state
        self._fb_active: bool = False
        self._fb_trade_dir: int = 0  # -1 SHORT (failed long), +1 LONG (failed short)
        self._fb_bars_remaining: int = 0
        self._fb_extreme: Decimal | None = None  # farthest close during fake breakout
        self._fb_orb_high: Decimal | None = None
        self._fb_orb_low: Decimal | None = None
        self._last_date: date | None = None

    # ── Strategy interface ─────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "FailedBreakout"

    def required_indicators(self) -> list[str]:
        return ["atr"]

    def evaluate(
        self,
        bar: Bar,
        indicators: IndicatorSnapshot,
        levels: LevelSnapshot,
        regime: RegimeDetector,
    ) -> Signal | None:
        """Evaluate current bar; return a reversal Signal when conditions fire.

        Volume history is updated on every call (before filtering) to keep the
        rolling average current even when the strategy is inactive.
        """
        avg_vol = self._avg_volume()
        self._volumes.append(bar.volume)

        bar_date = bar.timestamp.astimezone(_ET).date()
        bar_time = bar.timestamp.astimezone(_ET).time()

        # Reset intraday state on a new trading day
        if self._last_date != bar_date:
            self._last_date = bar_date
            self._fb_active = False

        # ── Hard filters ──────────────────────────────────────────────────────

        if bar_time >= _CUTOFF:
            return None

        if not levels.orb_complete:
            return None

        orb_high = levels.orb_high
        orb_low = levels.orb_low
        if orb_high is None or orb_low is None:
            return None

        vix = regime.vix_level
        adx = regime.adx_value
        if vix is None or adx is None:
            return None
        if vix >= _VIX_MAX:
            logger.debug("fb_filter_high_vix", vix=str(vix))
            return None
        if adx <= _ADX_MIN:
            logger.debug("fb_filter_low_adx", adx=str(adx))
            return None

        atr = indicators.atr
        if atr is None:
            return None

        close = bar.close

        # ── FB state machine ──────────────────────────────────────────────────

        if not self._fb_active:
            # Check whether this bar starts a fake-breakout watch
            fb_dir = self._detect_setup(close, orb_high, orb_low, atr)
            if fb_dir != 0:
                self._fb_active = True
                self._fb_trade_dir = fb_dir
                self._fb_bars_remaining = _REVERSAL_WINDOW
                self._fb_extreme = close
                self._fb_orb_high = orb_high
                self._fb_orb_low = orb_low
                logger.debug(
                    "fb_setup_detected",
                    direction="short" if fb_dir == -1 else "long",
                    close=str(close),
                    orb_high=str(orb_high),
                    orb_low=str(orb_low),
                )
            return None  # no entry on the setup bar itself

        else:
            # Update extreme (farthest point of the fake breakout)
            if self._fb_trade_dir == -1:
                self._fb_extreme = max(self._fb_extreme, close)  # type: ignore[type-var]
            else:
                self._fb_extreme = min(self._fb_extreme, close)  # type: ignore[type-var]

            # Expire the watch if countdown reached zero
            self._fb_bars_remaining -= 1
            if self._fb_bars_remaining <= 0:
                logger.debug("fb_watch_expired")
                self._fb_active = False
                return None

        # ── Reversal check ────────────────────────────────────────────────────

        assert self._fb_orb_high is not None
        assert self._fb_orb_low is not None
        assert self._fb_extreme is not None

        signal = self._check_reversal(
            bar=bar,
            close=close,
            avg_vol=avg_vol,
            atr=atr,
            orb_high=self._fb_orb_high,
            orb_low=self._fb_orb_low,
            fb_extreme=self._fb_extreme,
            regime=regime,
            indicators=indicators,
            levels=levels,
        )

        if signal is not None:
            self._fb_active = False

        return signal

    # ── helpers ────────────────────────────────────────────────────────────────

    def _avg_volume(self) -> Decimal | None:
        """Rolling average volume over the last ``_VOL_WINDOW`` bars."""
        if not self._volumes:
            return None
        return Decimal(str(sum(self._volumes) / len(self._volumes)))

    def _detect_setup(
        self,
        close: Decimal,
        orb_high: Decimal,
        orb_low: Decimal,
        atr: Decimal,
    ) -> int:
        """Return -1 (failed long setup), +1 (failed short setup), or 0."""
        min_ext = _ATR_MIN_EXT * atr
        max_ext = _ATR_MAX_EXT * atr
        above = close - orb_high
        if min_ext < above <= max_ext:
            return -1  # poked above ORB high → watch for SHORT entry
        below = orb_low - close
        if min_ext < below <= max_ext:
            return +1  # poked below ORB low → watch for LONG entry
        return 0

    def _check_reversal(
        self,
        bar: Bar,
        close: Decimal,
        avg_vol: Decimal | None,
        atr: Decimal,
        orb_high: Decimal,
        orb_low: Decimal,
        fb_extreme: Decimal,
        regime: RegimeDetector,
        indicators: IndicatorSnapshot,
        levels: LevelSnapshot,
    ) -> Signal | None:
        """Fire a signal if reversal conditions are fully satisfied."""
        if avg_vol is None:
            return None

        vol = Decimal(str(bar.volume))
        if vol < avg_vol * _VOL_MULTIPLIER:
            logger.debug("fb_filter_volume", volume=str(vol), avg=str(avg_vol))
            return None

        if self._fb_trade_dir == -1:
            # Failed LONG → go SHORT: price must be back below ORB high
            if close >= orb_high:
                return None
            stop = fb_extreme
            target = orb_low
            risk = stop - close
            if risk <= 0:
                return None
            rr = (close - target) / risk
            if rr < _MIN_RR:
                logger.debug("fb_filter_rr_short", rr=str(rr))
                return None
            direction = Direction.SHORT
            reason = (
                f"Failed long breakout: close {close} reversed below ORB high {orb_high} "
                f"(extreme={fb_extreme}) with volume {bar.volume:,}"
            )
        else:
            # Failed SHORT → go LONG: price must be back above ORB low
            if close <= orb_low:
                return None
            stop = fb_extreme
            target = orb_high
            risk = close - stop
            if risk <= 0:
                return None
            rr = (target - close) / risk
            if rr < _MIN_RR:
                logger.debug("fb_filter_rr_long", rr=str(rr))
                return None
            direction = Direction.LONG
            reason = (
                f"Failed short breakdown: close {close} reversed above ORB low {orb_low} "
                f"(extreme={fb_extreme}) with volume {bar.volume:,}"
            )

        signal = Signal(
            direction=direction,
            strategy_name=self.name,
            entry_price=close,
            stop_price=stop,
            target_price=target,
            risk_reward_ratio=rr,
            confidence_score=3,
            reason=reason,
            timeframe=bar.timeframe,
            regime=regime.current_regime,
            vix=regime.vix_level,
            adx=regime.adx_value,
            indicators_snapshot=indicators,
            levels_snapshot=levels,
            timestamp=bar.timestamp,
        )

        logger.info(
            "fb_signal",
            direction=str(direction),
            entry=str(close),
            stop=str(stop),
            target=str(target),
            rr=str(rr),
        )
        return signal
