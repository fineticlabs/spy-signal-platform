"""Tests for remaining structural fixes: realized vol, sector limits, signal scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.config import AppSettings, RiskSettings
from src.models import Direction, Regime, Signal, TimeFrame
from src.risk.cooldown import CooldownTracker
from src.risk.manager import SECTOR_MAP, RiskManager
from src.strategies.regime import RegimeDetector

_ET = ZoneInfo("America/New_York")


def _make_signal(symbol: str = "SPY", et_hour: int = 9, et_minute: int = 36) -> Signal:
    naive = datetime.strptime(f"2024-01-15 {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
    ts = naive.replace(tzinfo=_ET).astimezone(UTC)
    return Signal(
        symbol=symbol,
        direction=Direction.LONG,
        strategy_name="ORB-5min",
        entry_price=Decimal("486"),
        stop_price=Decimal("483"),
        target_price=Decimal("492"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=3,
        reason="test",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        vix=Decimal("18"),
        adx=Decimal("28"),
        timestamp=ts,
    )


# ── Realized vol filter tests ────────────────────────────────────────────────


class TestRealizedVolFilter:
    def test_regime_stores_realized_vol(self) -> None:
        regime = RegimeDetector()
        regime.set_realized_vol(Decimal("0.15"))
        assert regime.realized_vol == Decimal("0.15")

    def test_realized_vol_none_by_default(self) -> None:
        regime = RegimeDetector()
        assert regime.realized_vol is None

    def test_config_has_realized_vol_max(self) -> None:
        settings = AppSettings()
        assert settings.realized_vol_max == 0.18


# ── Sector limit tests ──────────────────────────────────────────────────────


class TestSectorLimits:
    def test_sector_map_covers_all_tickers(self) -> None:
        """All 26 default tickers have a sector assignment."""
        settings = AppSettings()
        for sym in settings.symbols:
            assert sym in SECTOR_MAP, f"{sym} missing from SECTOR_MAP"

    def test_semiconductor_tickers_in_same_sector(self) -> None:
        semi_tickers = ["AMD", "INTC", "MU", "MRVL", "SMCI", "SOXL", "ARM"]
        for sym in semi_tickers:
            assert SECTOR_MAP[sym] == "semi"

    def test_sector_limit_blocks_third_same_sector(self) -> None:
        """With 2 semi positions open, a 3rd semi signal is rejected."""
        cooldown = CooldownTracker()
        settings = RiskSettings(
            account_size=Decimal("200000"),
            risk_per_trade_pct=Decimal("1.0"),
            max_daily_loss_pct=Decimal("3.0"),
            max_trades_per_day=5,
            max_concurrent_positions=5,
            max_same_sector=2,
        )
        rm = RiskManager(
            cooldown=cooldown,
            settings=settings,
            get_position_count=lambda: 2,
            get_open_symbols=lambda: ["AMD", "INTC"],  # 2 semis open
        )
        decision = rm.approve(_make_signal(symbol="MU"))
        assert not decision.approved
        assert "sector" in decision.reason.lower()

    def test_different_sector_allowed(self) -> None:
        """Non-semi signal allowed when 2 semis are open."""
        cooldown = CooldownTracker()
        settings = RiskSettings(
            account_size=Decimal("200000"),
            risk_per_trade_pct=Decimal("1.0"),
            max_daily_loss_pct=Decimal("3.0"),
            max_trades_per_day=5,
            max_concurrent_positions=5,
            max_same_sector=2,
        )
        rm = RiskManager(
            cooldown=cooldown,
            settings=settings,
            get_position_count=lambda: 2,
            get_open_symbols=lambda: ["AMD", "INTC"],  # 2 semis open
        )
        decision = rm.approve(_make_signal(symbol="SPY"))  # index, not semi
        assert decision.approved

    def test_no_open_symbols_callback_skips_check(self) -> None:
        """Without callback, sector check is skipped."""
        cooldown = CooldownTracker()
        settings = RiskSettings(
            account_size=Decimal("200000"),
            risk_per_trade_pct=Decimal("1.0"),
            max_daily_loss_pct=Decimal("3.0"),
            max_trades_per_day=5,
            max_concurrent_positions=5,
            max_same_sector=2,
        )
        rm = RiskManager(cooldown=cooldown, settings=settings)
        decision = rm.approve(_make_signal(symbol="AMD"))
        assert decision.approved


# ── Signal scoring tests ─────────────────────────────────────────────────────


class TestSignalScoring:
    def test_score_higher_with_higher_adx(self) -> None:
        """Higher ADX produces higher score."""
        sig_low = _make_signal()
        sig_low = sig_low.model_copy(update={"adx": Decimal("20")})
        sig_high = _make_signal()
        sig_high = sig_high.model_copy(update={"adx": Decimal("40")})

        score_low = RiskManager.score_signal(sig_low)
        score_high = RiskManager.score_signal(sig_high)
        assert score_high > score_low

    def test_score_higher_with_higher_confidence(self) -> None:
        sig_low = _make_signal()
        sig_low = sig_low.model_copy(update={"confidence_score": 1})
        sig_high = _make_signal()
        sig_high = sig_high.model_copy(update={"confidence_score": 5})

        assert RiskManager.score_signal(sig_high) > RiskManager.score_signal(sig_low)

    def test_score_returns_float(self) -> None:
        score = RiskManager.score_signal(_make_signal())
        assert isinstance(score, float)
        assert 0 < score < 2
