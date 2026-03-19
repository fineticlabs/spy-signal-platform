"""Tests for src.main — registry, pipelines, cast_trades, _process_bar."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.main import _build_pipelines, _build_registry, _process_bar, _SymbolPipeline, cast_trades
from src.models import (
    Direction,
    IndicatorSnapshot,
    LevelSnapshot,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
    TradeResult,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_EXPECTED_INDICATORS = {"ema9", "ema20", "ema50", "rsi", "macd", "atr", "adx"}


def _make_raw_trade(**overrides: object) -> dict[str, object]:
    """Return a valid raw-trade dict, optionally overriding fields."""
    base: dict[str, object] = {
        "timestamp": "2024-01-15T14:36:00+00:00",
        "direction": "LONG",
        "strategy_name": "ORB",
        "entry_price": "481.50",
        "stop_price": "479.00",
        "target_price": "486.50",
        "pnl": "150.00",
    }
    base.update(overrides)
    return base


def _make_signal(**overrides: object) -> Signal:
    """Return a valid Signal for use in _process_bar mocks."""
    defaults: dict[str, object] = {
        "symbol": "SPY",
        "direction": Direction.LONG,
        "strategy_name": "ORB",
        "entry_price": Decimal("481.50"),
        "stop_price": Decimal("479.00"),
        "target_price": Decimal("486.50"),
        "risk_reward_ratio": Decimal("2.0"),
        "confidence_score": 3,
        "reason": "ORB long breakout",
        "timeframe": TimeFrame.ONE_MIN,
        "regime": Regime.TRENDING_UP,
        "timestamp": datetime(2024, 1, 15, 14, 36, tzinfo=UTC),
    }
    defaults.update(overrides)
    return Signal(**defaults)


def _make_mock_bar() -> MagicMock:
    """Return a lightweight mock Bar for _process_bar tests."""
    bar = MagicMock()
    bar.symbol = "SPY"
    bar.timestamp = datetime(2024, 1, 15, 14, 36, tzinfo=UTC)
    return bar


def _make_mock_pipeline() -> MagicMock:
    """Return a fully-mocked _SymbolPipeline for _process_bar tests."""
    pipeline = MagicMock()
    pipeline.registry.get_snapshot.return_value = IndicatorSnapshot()
    pipeline.levels.get_levels.return_value = LevelSnapshot()
    pipeline.strategy.evaluate.return_value = None
    pipeline.regime = MagicMock()
    return pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# _build_registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildRegistry:
    """Tests for _build_registry()."""

    def test_returns_indicator_registry(self) -> None:
        from src.indicators.registry import IndicatorRegistry

        registry = _build_registry()
        assert isinstance(registry, IndicatorRegistry)

    def test_has_seven_indicators(self) -> None:
        registry = _build_registry()
        assert len(registry) == 7
        # All 7 registered names map to IndicatorSnapshot fields
        snapshot_fields = set(type(registry.get_snapshot()).model_fields.keys())
        assert _EXPECTED_INDICATORS.issubset(snapshot_fields)

    def test_correct_indicator_names(self) -> None:
        registry = _build_registry()
        # Access internal _indicators dict to verify names
        names = set(registry._indicators.keys())
        assert names == _EXPECTED_INDICATORS


# ═══════════════════════════════════════════════════════════════════════════════
# _build_pipelines
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildPipelines:
    """Tests for _build_pipelines()."""

    def test_creates_one_pipeline_per_symbol(self) -> None:
        mock_db = MagicMock()
        symbols = ["SPY", "QQQ", "AAPL"]
        pipelines = _build_pipelines(symbols, mock_db)
        assert set(pipelines.keys()) == set(symbols)

    def test_each_pipeline_has_required_components(self) -> None:
        mock_db = MagicMock()
        pipelines = _build_pipelines(["SPY"], mock_db)
        p = pipelines["SPY"]
        assert p.registry is not None
        assert p.levels is not None
        assert p.regime is not None
        assert p.strategy is not None

    def test_single_symbol(self) -> None:
        mock_db = MagicMock()
        pipelines = _build_pipelines(["TSLA"], mock_db)
        assert len(pipelines) == 1
        assert "TSLA" in pipelines


# ═══════════════════════════════════════════════════════════════════════════════
# _SymbolPipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestSymbolPipeline:
    """Tests for _SymbolPipeline dataclass."""

    def test_stores_symbol_correctly(self) -> None:
        p = _SymbolPipeline(
            symbol="AAPL",
            registry=MagicMock(),
            levels=MagicMock(),
            regime=MagicMock(),
            strategy=MagicMock(),
        )
        assert p.symbol == "AAPL"


# ═══════════════════════════════════════════════════════════════════════════════
# cast_trades
# ═══════════════════════════════════════════════════════════════════════════════


class TestCastTrades:
    """Tests for cast_trades()."""

    def test_valid_dict_produces_trade_result(self) -> None:
        raw = [_make_raw_trade()]
        results = cast_trades(raw)
        assert len(results) == 1
        assert isinstance(results[0], TradeResult)
        assert results[0].pnl == Decimal("150.00")

    def test_tz_naive_timestamp_gets_utc(self) -> None:
        raw = [_make_raw_trade(timestamp="2024-01-15T14:36:00")]
        results = cast_trades(raw)
        assert len(results) == 1
        assert results[0].timestamp.tzinfo is not None

    def test_invalid_dict_skipped(self) -> None:
        raw = [{"bad_key": "bad_value"}]
        results = cast_trades(raw)
        assert results == []

    def test_empty_list_returns_empty(self) -> None:
        assert cast_trades([]) == []

    def test_multiple_dicts_correct_count(self) -> None:
        raw = [_make_raw_trade(pnl="100.00"), _make_raw_trade(pnl="-50.00")]
        results = cast_trades(raw)
        assert len(results) == 2

    def test_symbol_defaults_to_question_mark(self) -> None:
        raw = [_make_raw_trade()]
        results = cast_trades(raw)
        assert results[0].signal.symbol == "?"

    def test_symbol_preserved_when_present(self) -> None:
        raw = [_make_raw_trade(symbol="QQQ")]
        results = cast_trades(raw)
        assert results[0].signal.symbol == "QQQ"


# ═══════════════════════════════════════════════════════════════════════════════
# _process_bar (async)
# ═══════════════════════════════════════════════════════════════════════════════


class TestProcessBar:
    """Tests for _process_bar() — fully mocked pipeline components."""

    @pytest.mark.asyncio
    async def test_no_signal_returns_without_dispatch(self) -> None:
        bar = _make_mock_bar()
        pipeline = _make_mock_pipeline()
        risk = MagicMock()
        dispatcher = AsyncMock()
        db = MagicMock()

        await _process_bar(bar, pipeline, risk, dispatcher, db)

        risk.approve.assert_not_called()
        dispatcher.dispatch_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_signal_approved_dispatch_called(self) -> None:
        bar = _make_mock_bar()
        pipeline = _make_mock_pipeline()
        signal = _make_signal()
        pipeline.strategy.evaluate.return_value = signal

        decision = RiskDecision(approved=True, reason="ok", position_size=10)
        risk = MagicMock()
        risk.approve.return_value = decision

        dispatcher = AsyncMock()
        db = MagicMock()

        await _process_bar(bar, pipeline, risk, dispatcher, db)

        risk.approve.assert_called_once()
        dispatcher.dispatch_signal.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_signal_rejected_no_dispatch(self) -> None:
        bar = _make_mock_bar()
        pipeline = _make_mock_pipeline()
        signal = _make_signal()
        pipeline.strategy.evaluate.return_value = signal

        decision = RiskDecision(approved=False, reason="cooldown", position_size=0)
        risk = MagicMock()
        risk.approve.return_value = decision

        dispatcher = AsyncMock()
        db = MagicMock()

        await _process_bar(bar, pipeline, risk, dispatcher, db)

        risk.approve.assert_called_once()
        dispatcher.dispatch_signal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_signal_with_executor_submits_order(self) -> None:
        bar = _make_mock_bar()
        pipeline = _make_mock_pipeline()
        signal = _make_signal()
        pipeline.strategy.evaluate.return_value = signal

        decision = RiskDecision(approved=True, reason="ok", position_size=10)
        risk = MagicMock()
        risk.approve.return_value = decision

        dispatcher = AsyncMock()
        db = MagicMock()

        executor = MagicMock()
        executor.submit_bracket_order.return_value = MagicMock(id="order-123")

        await _process_bar(bar, pipeline, risk, dispatcher, db, executor=executor)

        executor.submit_bracket_order.assert_called_once()
        call_kwargs = executor.submit_bracket_order.call_args
        assert call_kwargs.kwargs["symbol"] == "SPY"
        assert call_kwargs.kwargs["direction"] == Direction.LONG

    @pytest.mark.asyncio
    async def test_persist_error_does_not_crash(self) -> None:
        bar = _make_mock_bar()
        pipeline = _make_mock_pipeline()
        signal = _make_signal()
        pipeline.strategy.evaluate.return_value = signal

        decision = RiskDecision(approved=True, reason="ok", position_size=10)
        risk = MagicMock()
        risk.approve.return_value = decision

        dispatcher = AsyncMock()
        db = MagicMock()

        with patch("src.main.insert_signal", side_effect=RuntimeError("DB write failed")):
            # Should not raise — error is caught internally
            await _process_bar(bar, pipeline, risk, dispatcher, db)

        # Dispatch still happens even if persist fails
        dispatcher.dispatch_signal.assert_awaited_once()
