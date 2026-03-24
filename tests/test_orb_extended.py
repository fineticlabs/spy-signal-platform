"""Extended tests for ORBStrategy covering all uncovered branches."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

from src.models import (
    Bar,
    Direction,
    IndicatorSnapshot,
    LevelSnapshot,
    Regime,
    TimeFrame,
)
from src.strategies.orb import ORBStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockRegime:
    """Minimal regime detector stub."""

    def __init__(
        self,
        vix: Decimal = Decimal("18"),
        adx: Decimal = Decimal("28"),
        regime: Regime = Regime.TRENDING_UP,
    ) -> None:
        self.vix_level = vix
        self.adx_value = adx
        self.current_regime = regime
        self.is_tradeable = True
        self.ema20_15m: Decimal | None = None  # matches RegimeDetector
        self.realized_vol: Decimal | None = None  # matches RegimeDetector


def _make_bar(
    close: Decimal,
    *,
    symbol: str = "AAPL",
    volume: int = 500_000,
    open_: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
    hour: int = 10,
    minute: int = 0,
) -> Bar:
    """Build a bar at the given ET hour/minute (internally stored as UTC)."""
    # ET offset is UTC-4 (EDT) for typical trading days
    ts = datetime(2025, 6, 10, hour + 4, minute, tzinfo=UTC)
    open_p = open_ if open_ is not None else close - Decimal("0.20")
    high_p = high if high is not None else close + Decimal("0.10")
    low_p = low if low is not None else close - Decimal("0.30")
    return Bar(
        symbol=symbol,
        timeframe=TimeFrame.FIVE_MIN,
        timestamp=ts,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close,
        volume=volume,
        vwap=close,
    )


def _levels(
    orb_high: Decimal = Decimal("450.00"),
    orb_low: Decimal = Decimal("448.00"),
    orb_complete: bool = True,
    rvol: Decimal | None = None,
) -> LevelSnapshot:
    return LevelSnapshot(
        orb_high=orb_high,
        orb_low=orb_low,
        orb_complete=orb_complete,
        rvol=rvol,
    )


def _indicators(atr: Decimal = Decimal("1.00")) -> IndicatorSnapshot:
    return IndicatorSnapshot(atr=atr)


def _prime_strategy(strategy: ORBStrategy, *, count: int = 21, volume: int = 100_000) -> None:
    """Feed enough bars to fill the volume window (20 bars)."""
    for i in range(count):
        bar = _make_bar(Decimal("449.00"), volume=volume, hour=9, minute=35 + i % 25)
        strategy._volumes.append(bar.volume)


_ECON_PATCH = "src.strategies.orb.is_high_impact_day"
_EARN_PATCH = "src.strategies.orb.is_earnings_blackout"


def _evaluate_long(
    strategy: ORBStrategy,
    *,
    close: Decimal = Decimal("451.00"),
    volume: int = 500_000,
    symbol: str = "AAPL",
    regime: MockRegime | None = None,
    levels: LevelSnapshot | None = None,
    indicators: IndicatorSnapshot | None = None,
    hour: int = 10,
    minute: int = 0,
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
    open_: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
):
    """Helper to call evaluate with 2-bar confirmation for a LONG breakout.

    Sends two consecutive bars at (hour, minute) and (hour, minute+1)
    to satisfy 2-bar confirmation. Returns the signal from the second bar.
    """
    _ind = indicators or _indicators()
    _lvl = levels or _levels()
    _reg = regime or MockRegime()
    kwargs = dict(
        spy_vwap=spy_vwap,
        spy_price=spy_price,
        vp_poc=vp_poc,
        vp_vah=vp_vah,
        vp_val=vp_val,
        vp_hvn=vp_hvn,
        vp_lvn=vp_lvn,
        vix_term_ratio=vix_term_ratio,
        hmm_regime=hmm_regime,
        kalman_stop_mult=kalman_stop_mult,
    )

    # First bar — sets pending confirmation
    bar1 = _make_bar(
        close,
        symbol=symbol,
        volume=volume,
        hour=hour,
        minute=minute,
        open_=open_,
        high=high,
        low=low,
    )
    strategy.evaluate(bar1, _ind, _lvl, _reg, **kwargs)

    # Second bar — confirms breakout
    bar2 = _make_bar(
        close,
        symbol=symbol,
        volume=volume,
        hour=hour,
        minute=minute + 1,
        open_=open_,
        high=high,
        low=low,
    )
    return strategy.evaluate(bar2, _ind, _lvl, _reg, **kwargs)


# ===========================================================================
# Test classes
# ===========================================================================


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestEconomicCalendarFilter:
    """1. Economic calendar filter blocks signal on high-impact days."""

    def test_high_impact_day_blocks_signal(self, _mock_econ, _mock_earn):
        _mock_econ.return_value = True  # override to True
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy)
        assert sig is None

    def test_non_high_impact_day_allows_signal(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy)
        assert sig is not None
        assert sig.direction == Direction.LONG


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestEarningsBlackout:
    """2. Earnings blackout adds EARNINGS tag (does not block)."""

    def test_earnings_blackout_adds_tag(self, _mock_econ, _mock_earn):
        _mock_earn.return_value = True
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy)
        assert sig is not None
        assert "EARNINGS" in sig.tags

    def test_no_earnings_no_tag(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy)
        assert sig is not None
        assert "EARNINGS" not in sig.tags


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestMarketDirectionConfirmation:
    """3. SPY VWAP alignment tags."""

    def test_spy_aligned_long_above_vwap(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            spy_vwap=Decimal("445.00"),
            spy_price=Decimal("446.00"),
        )
        assert sig is not None
        assert "SPY_ALIGNED" in sig.tags
        assert sig.confidence_score >= 4  # base 3 + 1

    def test_spy_conflict_long_below_vwap(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            spy_vwap=Decimal("446.00"),
            spy_price=Decimal("444.00"),
        )
        assert sig is not None
        assert "SPY_CONFLICT" in sig.tags
        assert sig.confidence_score == 1  # base 3 - 2

    def test_spy_aligned_short_below_vwap(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            close=Decimal("447.00"),
            spy_vwap=Decimal("446.00"),
            spy_price=Decimal("444.00"),
        )
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert "SPY_ALIGNED" in sig.tags

    def test_spy_conflict_short_above_vwap(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            close=Decimal("447.00"),
            spy_vwap=Decimal("444.00"),
            spy_price=Decimal("446.00"),
        )
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert "SPY_CONFLICT" in sig.tags

    def test_spy_exempt_ticker_no_tag(self, _mock_econ, _mock_earn):
        """SPY itself is exempt from market direction check."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            symbol="SPY",
            spy_vwap=Decimal("445.00"),
            spy_price=Decimal("444.00"),
        )
        assert sig is not None
        assert "SPY_ALIGNED" not in sig.tags
        assert "SPY_CONFLICT" not in sig.tags

    def test_no_spy_data_no_tag(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, spy_vwap=None, spy_price=None)
        assert sig is not None
        assert "SPY_ALIGNED" not in sig.tags
        assert "SPY_CONFLICT" not in sig.tags


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestRvolConfidenceAdjustment:
    """4. RVOL confidence adjustment."""

    def test_low_rvol_demotes_confidence(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, levels=_levels(rvol=Decimal("0.3")))
        assert sig is not None
        assert "LOW_RVOL" in sig.tags
        assert sig.confidence_score == 1  # base 3 - 2

    def test_high_rvol_boosts_confidence(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, levels=_levels(rvol=Decimal("2.0")))
        assert sig is not None
        assert "HIGH_RVOL" in sig.tags
        assert sig.confidence_score == 4  # base 3 + 1

    def test_medium_rvol_no_adjustment(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, levels=_levels(rvol=Decimal("1.0")))
        assert sig is not None
        assert "LOW_RVOL" not in sig.tags
        assert "HIGH_RVOL" not in sig.tags
        assert sig.confidence_score == 3

    def test_rvol_none_no_adjustment(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, levels=_levels(rvol=None))
        assert sig is not None
        assert "LOW_RVOL" not in sig.tags
        assert "HIGH_RVOL" not in sig.tags

    def test_rvol_exactly_half_demotes(self, _mock_econ, _mock_earn):
        """RVOL < 0.5 threshold (strict less-than)."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, levels=_levels(rvol=Decimal("0.5")))
        assert sig is not None
        # rvol=0.5 is NOT < 0.5, so no LOW_RVOL
        assert "LOW_RVOL" not in sig.tags

    def test_rvol_exactly_one_point_five_boosts(self, _mock_econ, _mock_earn):
        """RVOL >= 1.5 threshold (gte)."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, levels=_levels(rvol=Decimal("1.5")))
        assert sig is not None
        assert "HIGH_RVOL" in sig.tags


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestVolumeProfileTags:
    """5. Volume profile tags."""

    def test_lvn_target_outside_value_area_high(self, _mock_econ, _mock_earn):
        """LONG target above VAH → VP_LVN_TARGET."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            vp_poc=Decimal("449.00"),
            vp_vah=Decimal("450.50"),
            vp_val=Decimal("447.50"),
        )
        assert sig is not None
        # target = entry(451) + 2 * 1.5 * 1.0 = 454.0, which is > vah(450.50)
        assert "VP_LVN_TARGET" in sig.tags

    def test_hvn_target_inside_value_area(self, _mock_econ, _mock_earn):
        """Target inside Value Area → VP_HVN_TARGET."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            vp_poc=Decimal("449.00"),
            vp_vah=Decimal("460.00"),
            vp_val=Decimal("440.00"),
        )
        assert sig is not None
        # target ~454 is inside [440, 460]
        assert "VP_HVN_TARGET" in sig.tags

    def test_poc_cross_long(self, _mock_econ, _mock_earn):
        """Entry-to-target path crosses POC → VP_POC_CROSS."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            vp_poc=Decimal("452.00"),
            vp_vah=Decimal("450.50"),
            vp_val=Decimal("447.50"),
        )
        assert sig is not None
        # close=451, poc=452, target=454 → close < poc < target
        assert "VP_POC_CROSS" in sig.tags

    def test_poc_cross_short(self, _mock_econ, _mock_earn):
        """SHORT: target-to-entry path crosses POC → VP_POC_CROSS."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            close=Decimal("447.00"),
            vp_poc=Decimal("446.00"),
            vp_vah=Decimal("450.00"),
            vp_val=Decimal("440.00"),
        )
        assert sig is not None
        assert sig.direction == Direction.SHORT
        # target = 447 - 2*1.5*1 = 444, poc=446, target < poc < close
        assert "VP_POC_CROSS" in sig.tags

    def test_no_vp_data_no_tags(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy)
        assert sig is not None
        assert "VP_LVN_TARGET" not in sig.tags
        assert "VP_HVN_TARGET" not in sig.tags
        assert "VP_POC_CROSS" not in sig.tags


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestVixTermStructure:
    """6. VIX term structure tags."""

    def test_contango_boosts_confidence(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, vix_term_ratio=Decimal("0.80"))
        assert sig is not None
        assert "CONTANGO" in sig.tags
        assert sig.confidence_score == 4  # base 3 + 1

    def test_backwardation_blocks_signal(self, _mock_econ, _mock_earn):
        """VIX backwardation is now a hard block (matches backtest)."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, vix_term_ratio=Decimal("1.10"))
        assert sig is None  # hard block, not just a tag

    def test_neutral_vix_ratio_no_tag(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, vix_term_ratio=Decimal("0.92"))
        assert sig is not None
        assert "CONTANGO" not in sig.tags
        assert "BACKWARDATION" not in sig.tags

    def test_no_vix_ratio_no_tag(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, vix_term_ratio=None)
        assert sig is not None
        assert "CONTANGO" not in sig.tags
        assert "BACKWARDATION" not in sig.tags


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestHmmRegime:
    """7. HMM regime tags."""

    def test_volatile_boosts_confidence(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, hmm_regime="VOLATILE")
        assert sig is not None
        assert "HMM_VOLATILE" in sig.tags
        assert sig.confidence_score == 4

    def test_calm_demotes_confidence(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, hmm_regime="CALM")
        assert sig is not None
        assert "HMM_CALM" in sig.tags
        assert sig.confidence_score == 2

    def test_normal_tag_only(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, hmm_regime="NORMAL")
        assert sig is not None
        assert "HMM_NORMAL" in sig.tags
        assert sig.confidence_score == 3  # unchanged

    def test_no_hmm_regime_no_tag(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, hmm_regime=None)
        assert sig is not None
        assert "HMM_VOLATILE" not in sig.tags
        assert "HMM_CALM" not in sig.tags
        assert "HMM_NORMAL" not in sig.tags


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestEngulfingBar:
    """8. Engulfing bar detection."""

    def test_engulfing_long_boosts_confidence(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        # Feed a small prior bar so the breakout bar engulfs it
        prev = _make_bar(
            Decimal("449.50"),
            open_=Decimal("449.40"),
            high=Decimal("449.60"),
            low=Decimal("449.30"),
            hour=9,
            minute=55,
        )
        strategy._recent_bars.append(prev)
        # Breakout bar: large green candle that engulfs prior body
        sig = _evaluate_long(
            strategy,
            close=Decimal("451.00"),
            open_=Decimal("449.20"),
            high=Decimal("451.20"),
            low=Decimal("449.10"),
        )
        assert sig is not None
        assert "ENGULFING" in sig.tags

    def test_no_engulfing_no_tag(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        # Prior bar is large, so breakout bar cannot engulf it
        prev = _make_bar(
            Decimal("449.50"),
            open_=Decimal("440.00"),
            high=Decimal("455.00"),
            low=Decimal("439.00"),
            hour=9,
            minute=55,
        )
        strategy._recent_bars.append(prev)
        sig = _evaluate_long(strategy)
        assert sig is not None
        assert "ENGULFING" not in sig.tags


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestInsideBarCompression:
    """9. Inside bar compression."""

    def test_compressed_bars_boost_confidence(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        # Build 3 prior bars where bar[1] is inside bar[0].
        # _evaluate_long adds 2 bars (2-bar confirmation), so 3 + 2 = 5
        # which fills the deque (maxlen=5) for compression check.
        bars = [
            _make_bar(
                Decimal("449.50"), high=Decimal("450.50"), low=Decimal("448.50"), hour=9, minute=50
            ),
            _make_bar(
                Decimal("449.50"), high=Decimal("450.00"), low=Decimal("449.00"), hour=9, minute=51
            ),  # inside
            _make_bar(
                Decimal("449.50"), high=Decimal("450.20"), low=Decimal("448.80"), hour=9, minute=52
            ),
        ]
        for b in bars:
            strategy._recent_bars.append(b)
        sig = _evaluate_long(strategy)
        assert sig is not None
        assert "COMPRESSED" in sig.tags

    def test_no_compression_no_tag(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        # Clear recent bars — with only 2 bars from _evaluate_long,
        # the compression check (requires >= 5 bars) won't fire
        strategy._recent_bars.clear()
        sig = _evaluate_long(strategy)
        assert sig is not None
        assert "COMPRESSED" not in sig.tags


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestKalmanAdaptiveStop:
    """10. Kalman adaptive stop multiplier."""

    def test_kalman_mult_widens_stop(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig_default = _evaluate_long(strategy)

        strategy2 = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy2)
        sig_wide = _evaluate_long(strategy2, kalman_stop_mult=Decimal("1.5"))

        assert sig_default is not None
        assert sig_wide is not None
        # Wider stop means lower stop for LONG
        assert sig_wide.stop_price < sig_default.stop_price
        # Target should be the same (decoupled from kalman)
        assert sig_wide.target_price == sig_default.target_price

    def test_kalman_mult_tightens_stop(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, kalman_stop_mult=Decimal("0.9"))
        assert sig is not None
        # stop = 451 - 1.5 * 1.0 * 0.9 = 451 - 1.35 = 449.65
        assert sig.stop_price == Decimal("451.00") - Decimal("1.35")

    def test_kalman_none_defaults_to_one(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, kalman_stop_mult=None)
        assert sig is not None
        # stop = 451 - 1.5 * 1.0 * 1.0 = 449.5
        assert sig.stop_price == Decimal("449.50")


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestConfidenceClamping:
    """11. Confidence clamped at min 1, max 5."""

    def test_confidence_clamped_at_max_5(self, _mock_econ, _mock_earn):
        """Stack multiple boosters: SPY_ALIGNED(+1) + HIGH_RVOL(+1) + CONTANGO(+1) + VOLATILE(+1) = 3+4=7 → 5."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            spy_vwap=Decimal("445.00"),
            spy_price=Decimal("446.00"),
            levels=_levels(rvol=Decimal("2.0")),
            vix_term_ratio=Decimal("0.80"),
            hmm_regime="VOLATILE",
        )
        assert sig is not None
        assert sig.confidence_score == 5

    def test_confidence_clamped_at_min_1(self, _mock_econ, _mock_earn):
        """Stack demoters: SPY_CONFLICT(-2) + LOW_RVOL(-2) + HMM_CALM(-1) = 3-5=-2 → 1."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(
            strategy,
            spy_vwap=Decimal("446.00"),
            spy_price=Decimal("444.00"),
            levels=_levels(rvol=Decimal("0.3")),
            hmm_regime="CALM",
        )
        assert sig is not None
        # Pydantic Signal has ge=1, so if code clamps correctly it passes
        assert sig.confidence_score == 1


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestRiskZeroReturnsNone:
    """12. Risk = 0 check returns None."""

    def test_zero_risk_returns_none(self, _mock_econ, _mock_earn):
        """If entry == stop (risk=0) signal should be None."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        # ATR=0 → stop = entry, risk = 0
        sig = _evaluate_long(strategy, indicators=_indicators(atr=Decimal("0")))
        assert sig is None

    def test_negative_risk_returns_none(self, _mock_econ, _mock_earn):
        """Negative ATR (should not happen, but guard works)."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, indicators=_indicators(atr=Decimal("-1.0")))
        assert sig is None


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestShortDirection:
    """Ensure SHORT direction works for key branches."""

    def test_short_signal_basic(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, close=Decimal("447.00"))
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert sig.stop_price > sig.entry_price  # stop above entry for SHORT
        assert sig.target_price < sig.entry_price

    def test_short_kalman_widens_stop(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, close=Decimal("447.00"), kalman_stop_mult=Decimal("1.5"))
        assert sig is not None
        assert sig.direction == Direction.SHORT
        # stop = 447 + 1.5 * 1.0 * 1.5 = 449.25
        assert sig.stop_price == Decimal("447.00") + Decimal("2.25")


@patch(_EARN_PATCH, return_value=False)
@patch(_ECON_PATCH, return_value=False)
class TestPriceInsideOrbRange:
    """Price inside ORB range returns None."""

    def test_price_inside_orb_returns_none(self, _mock_econ, _mock_earn):
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        _prime_strategy(strategy)
        sig = _evaluate_long(strategy, close=Decimal("449.00"))
        assert sig is None
