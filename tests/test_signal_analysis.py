"""Tests for signal quality analysis and execution comparison in EOD report.

Covers:
- _evaluate_signal_outcomes (LONG/SHORT winner/loser/open, edge cases)
- update_signal_outcome DB function
- EOD status message includes signal quality section
- Bar cache trimming to _BAR_CACHE_MAX
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.main import (
    _BAR_CACHE_MAX,
    _evaluate_signal_outcomes,
    _process_bar,
    _send_eod_status,
    _SymbolPipeline,
)
from src.models import IndicatorSnapshot, LevelSnapshot
from src.storage.queries import ensure_schema, update_signal_outcome
from tests.conftest import make_bar, make_utc_ts

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_approved_signal_row(
    *,
    signal_id: int = 1,
    direction: str = "LONG",
    entry_price: str = "480.00",
    stop_price: str = "478.00",
    target_price: str = "484.00",
    position_size: int = 100,
    timestamp: str = "2024-01-15T14:36:00+00:00",
    symbol: str = "SPY",
) -> dict[str, object]:
    """Return a dict mimicking a row from ``query_recent_signals``."""
    return {
        "id": signal_id,
        "approved": 1,
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "position_size": position_size,
        "timestamp": timestamp,
        "symbol": symbol,
        "strategy_name": "ORB",
        "reject_reason": "",
    }


def _make_pipeline_with_cache(
    symbol: str = "SPY",
    bars: list | None = None,
) -> _SymbolPipeline:
    """Build a mock _SymbolPipeline with the given bar_cache."""
    pipeline = MagicMock(spec=_SymbolPipeline)
    pipeline.symbol = symbol
    pipeline.bar_cache = bars or []
    return pipeline


def _make_bar_at(
    ts: datetime,
    *,
    high: Decimal = Decimal("482.00"),
    low: Decimal = Decimal("479.00"),
) -> MagicMock:
    """Return a Bar-like object at the given timestamp."""
    return make_bar(timestamp=ts, high=high, low=low)


# ═══════════════════════════════════════════════════════════════════════════════
# _evaluate_signal_outcomes
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateSignalOutcomesLongWinner:
    """LONG signal where a bar hits the target first."""

    def test_long_winner(self) -> None:
        sig = _make_approved_signal_row(
            entry_price="480.00",
            stop_price="478.00",
            target_price="484.00",
            position_size=100,
        )
        # Bar after signal with high >= target
        bar = _make_bar_at(
            make_utc_ts(2024, 1, 15, 14, 40),
            high=Decimal("485.00"),
            low=Decimal("479.50"),
        )
        pipeline = _make_pipeline_with_cache("SPY", [bar])
        db = MagicMock()
        db.conn = MagicMock()

        with (
            patch("src.main.query_recent_signals", return_value=[sig]),
            patch("src.main.update_signal_outcome") as mock_update,
        ):
            results = _evaluate_signal_outcomes(db, {"SPY": pipeline})

        assert len(results["winners"]) == 1
        assert results["winners"][0]["outcome"] == "winner"
        assert results["winners"][0]["theoretical_pnl"] == 400.0  # (484-480)*100
        mock_update.assert_called_once_with(db.conn, 1, "winner")


class TestEvaluateSignalOutcomesShortWinner:
    """SHORT signal where a bar hits the target first."""

    def test_short_winner(self) -> None:
        sig = _make_approved_signal_row(
            direction="SHORT",
            entry_price="480.00",
            stop_price="482.00",
            target_price="476.00",
            position_size=50,
        )
        bar = _make_bar_at(
            make_utc_ts(2024, 1, 15, 14, 40),
            high=Decimal("480.50"),
            low=Decimal("475.00"),
        )
        pipeline = _make_pipeline_with_cache("SPY", [bar])
        db = MagicMock()
        db.conn = MagicMock()

        with (
            patch("src.main.query_recent_signals", return_value=[sig]),
            patch("src.main.update_signal_outcome"),
        ):
            results = _evaluate_signal_outcomes(db, {"SPY": pipeline})

        assert len(results["winners"]) == 1
        assert results["winners"][0]["theoretical_pnl"] == 200.0  # (480-476)*50


class TestEvaluateSignalOutcomesLongLoser:
    """LONG signal where a bar hits the stop first."""

    def test_long_loser(self) -> None:
        sig = _make_approved_signal_row(
            entry_price="480.00",
            stop_price="478.00",
            target_price="484.00",
            position_size=100,
        )
        bar = _make_bar_at(
            make_utc_ts(2024, 1, 15, 14, 40),
            high=Decimal("480.50"),
            low=Decimal("477.00"),
        )
        pipeline = _make_pipeline_with_cache("SPY", [bar])
        db = MagicMock()
        db.conn = MagicMock()

        with (
            patch("src.main.query_recent_signals", return_value=[sig]),
            patch("src.main.update_signal_outcome"),
        ):
            results = _evaluate_signal_outcomes(db, {"SPY": pipeline})

        assert len(results["losers"]) == 1
        assert results["losers"][0]["outcome"] == "loser"
        assert results["losers"][0]["theoretical_pnl"] == -200.0  # (478-480)*100


class TestEvaluateSignalOutcomesOpen:
    """Neither target nor stop hit — stays open."""

    def test_open(self) -> None:
        sig = _make_approved_signal_row(
            entry_price="480.00",
            stop_price="478.00",
            target_price="484.00",
            position_size=100,
        )
        # Bar that doesn't breach either level
        bar = _make_bar_at(
            make_utc_ts(2024, 1, 15, 14, 40),
            high=Decimal("481.00"),
            low=Decimal("479.00"),
        )
        pipeline = _make_pipeline_with_cache("SPY", [bar])
        db = MagicMock()
        db.conn = MagicMock()

        with (
            patch("src.main.query_recent_signals", return_value=[sig]),
            patch("src.main.update_signal_outcome"),
        ):
            results = _evaluate_signal_outcomes(db, {"SPY": pipeline})

        assert len(results["open"]) == 1
        assert results["open"][0]["outcome"] == "open"


class TestEvaluateSignalOutcomesNoSignals:
    """No approved signals today — empty results."""

    def test_no_signals(self) -> None:
        db = MagicMock()
        db.conn = MagicMock()

        with patch("src.main.query_recent_signals", return_value=[]):
            results = _evaluate_signal_outcomes(db, {"SPY": _make_pipeline_with_cache()})

        assert results == {"winners": [], "losers": [], "open": [], "no_data": []}


class TestEvaluateSignalOutcomesNoBars:
    """Approved signals but no cached bars — all open."""

    def test_no_bars_marks_no_data(self) -> None:
        """Empty cache + REST fetch returns nothing → no_data, not open."""
        sig = _make_approved_signal_row()
        pipeline = _make_pipeline_with_cache("SPY", [])
        db = MagicMock()
        db.conn = MagicMock()

        with (
            patch("src.main.query_recent_signals", return_value=[sig]),
            patch("src.main.update_signal_outcome"),
            patch("src.main._fetch_bars_for_signal", return_value=[]),
        ):
            results = _evaluate_signal_outcomes(db, {"SPY": pipeline})

        assert len(results["no_data"]) == 1
        assert results["no_data"][0]["outcome"] == "no_data"
        assert len(results["open"]) == 0


class TestEvaluateSignalOutcomesFallbackPipeline:
    """Signal symbol not in pipelines — falls back to first pipeline."""

    def test_fallback_pipeline(self) -> None:
        sig = _make_approved_signal_row(symbol="AAPL")
        bar = _make_bar_at(
            make_utc_ts(2024, 1, 15, 14, 40),
            high=Decimal("485.00"),
            low=Decimal("479.50"),
        )
        pipeline = _make_pipeline_with_cache("SPY", [bar])
        db = MagicMock()
        db.conn = MagicMock()

        with (
            patch("src.main.query_recent_signals", return_value=[sig]),
            patch("src.main.update_signal_outcome"),
        ):
            results = _evaluate_signal_outcomes(db, {"SPY": pipeline})

        # Should still evaluate using SPY pipeline as fallback
        assert len(results["winners"]) == 1


class TestEvaluateSignalOutcomesShortLoser:
    """SHORT signal where stop is hit."""

    def test_short_loser(self) -> None:
        sig = _make_approved_signal_row(
            direction="SHORT",
            entry_price="480.00",
            stop_price="482.00",
            target_price="476.00",
            position_size=50,
        )
        bar = _make_bar_at(
            make_utc_ts(2024, 1, 15, 14, 40),
            high=Decimal("483.00"),
            low=Decimal("479.00"),
        )
        pipeline = _make_pipeline_with_cache("SPY", [bar])
        db = MagicMock()
        db.conn = MagicMock()

        with (
            patch("src.main.query_recent_signals", return_value=[sig]),
            patch("src.main.update_signal_outcome"),
        ):
            results = _evaluate_signal_outcomes(db, {"SPY": pipeline})

        assert len(results["losers"]) == 1
        assert results["losers"][0]["theoretical_pnl"] == -100.0  # (480-482)*50


# ═══════════════════════════════════════════════════════════════════════════════
# update_signal_outcome (DB)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpdateSignalOutcome:
    """Tests for update_signal_outcome DB function."""

    def test_updates_outcome_column(self, tmp_path: object) -> None:
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))  # type: ignore[operator]
        db.connect()
        ensure_schema(db.conn)

        # Insert a signal row manually
        db.conn.execute(
            """
            INSERT INTO signals
                (timestamp, strategy_name, direction, entry_price, stop_price,
                 target_price, risk_reward, confidence, reason, timeframe, regime,
                 approved, position_size, reject_reason, symbol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2024-01-15T14:36:00+00:00",
                "ORB",
                "LONG",
                "480.00",
                "478.00",
                "484.00",
                "2.0",
                3,
                "test",
                "1Min",
                "TRENDING_UP",
                1,
                100,
                "",
                "SPY",
            ),
        )
        db.conn.commit()

        update_signal_outcome(db.conn, 1, "winner")

        row = db.conn.execute("SELECT outcome FROM signals WHERE id = 1").fetchone()
        assert row["outcome"] == "winner"

        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# EOD status includes signal quality
# ═══════════════════════════════════════════════════════════════════════════════


class TestEodStatusIncludesSignalQuality:
    """Test that _send_eod_status includes signal quality and execution sections."""

    @pytest.mark.asyncio
    async def test_eod_includes_signal_quality(self) -> None:
        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()

        pipeline = MagicMock()
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}

        signals = [
            {"approved": 1, "reject_reason": ""},
            {"approved": 1, "reject_reason": ""},
            {"approved": 0, "reject_reason": "cooldown"},
        ]
        trades = [{"pnl": "150.00"}]

        outcomes = {
            "winners": [{"theoretical_pnl": 400.0}],
            "losers": [{"theoretical_pnl": -200.0}],
            "open": [],
            "no_data": [],
        }

        with (
            patch("src.main.query_recent_signals", return_value=signals),
            patch("src.main.query_recent_trades", return_value=trades),
            patch("src.main._evaluate_signal_outcomes", return_value=outcomes),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "Signal Quality" in msg
        assert "Win rate: 50%" in msg
        assert "Theoretical P&L: $+200.00" in msg

    @pytest.mark.asyncio
    async def test_eod_includes_execution_comparison(self) -> None:
        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()

        pipeline = MagicMock()
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}

        signals = [
            {"approved": 1, "reject_reason": ""},
            {"approved": 1, "reject_reason": ""},
        ]
        trades = [{"pnl": "150.00"}]

        outcomes = {"winners": [], "losers": [], "open": [], "no_data": []}

        with (
            patch("src.main.query_recent_signals", return_value=signals),
            patch("src.main.query_recent_trades", return_value=trades),
            patch("src.main._evaluate_signal_outcomes", return_value=outcomes),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "Execution" in msg
        assert "Executed: 1/2 signals" in msg
        assert "Skipped: 1" in msg
        assert "Actual P&L: $+150.00" in msg

    @pytest.mark.asyncio
    async def test_eod_open_signals_shown(self) -> None:
        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()

        pipeline = MagicMock()
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}

        signals = [{"approved": 1, "reject_reason": ""}]
        trades: list[dict[str, object]] = []

        outcomes = {
            "winners": [],
            "losers": [],
            "open": [{"theoretical_pnl": 0.0}],
            "no_data": [],
        }

        with (
            patch("src.main.query_recent_signals", return_value=signals),
            patch("src.main.query_recent_trades", return_value=trades),
            patch("src.main._evaluate_signal_outcomes", return_value=outcomes),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "Still open at close: 1" in msg

    @pytest.mark.asyncio
    async def test_eod_no_data_signals_shown(self) -> None:
        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()

        pipeline = MagicMock()
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}

        signals = [{"approved": 1, "reject_reason": ""}]
        trades: list[dict[str, object]] = []

        outcomes = {
            "winners": [],
            "losers": [],
            "open": [],
            "no_data": [{"theoretical_pnl": 0.0}, {"theoretical_pnl": 0.0}],
        }

        with (
            patch("src.main.query_recent_signals", return_value=signals),
            patch("src.main.query_recent_trades", return_value=trades),
            patch("src.main._evaluate_signal_outcomes", return_value=outcomes),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "No data: 2" in msg
        assert "cache empty after restart" in msg


# ═══════════════════════════════════════════════════════════════════════════════
# Bar cache trimming
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarCacheTrimmed:
    """Test that bar_cache stays under _BAR_CACHE_MAX."""

    @pytest.mark.asyncio
    async def test_cache_trimmed_at_max(self) -> None:
        pipeline = MagicMock()
        pipeline.registry.get_snapshot.return_value = IndicatorSnapshot()
        pipeline.levels.get_levels.return_value = LevelSnapshot()
        pipeline.strategy.evaluate.return_value = None
        pipeline.regime = MagicMock()

        # Pre-fill cache to exactly _BAR_CACHE_MAX
        pipeline.bar_cache = [make_bar() for _ in range(_BAR_CACHE_MAX)]

        bar = make_bar(symbol="SPY")
        risk = MagicMock()
        risk.approve.return_value = MagicMock(approved=False, reason="test")

        await _process_bar(bar, pipeline, risk, AsyncMock(), MagicMock())

        # After appending one more, it should be trimmed back to _BAR_CACHE_MAX
        assert len(pipeline.bar_cache) == _BAR_CACHE_MAX

    def test_bar_cache_max_constant(self) -> None:
        """_BAR_CACHE_MAX is set to 400."""
        assert _BAR_CACHE_MAX == 400

    def test_pipeline_bar_cache_default_empty(self) -> None:
        """New _SymbolPipeline starts with empty bar_cache."""
        p = _SymbolPipeline(
            symbol="SPY",
            registry=MagicMock(),
            levels=MagicMock(),
            regime=MagicMock(),
            strategy=MagicMock(),
        )
        assert p.bar_cache == []


class TestEvaluateSignalOutcomesInvalidPrices:
    """Signals with zero prices are skipped."""

    def test_zero_entry_skipped(self) -> None:
        sig = _make_approved_signal_row(entry_price="0")
        db = MagicMock()
        db.conn = MagicMock()

        with patch("src.main.query_recent_signals", return_value=[sig]):
            results = _evaluate_signal_outcomes(db, {"SPY": _make_pipeline_with_cache()})

        assert results == {"winners": [], "losers": [], "open": [], "no_data": []}


class TestEvaluateSignalOutcomesNoPipelines:
    """No pipelines at all — should not crash."""

    def test_empty_pipelines(self) -> None:
        sig = _make_approved_signal_row()
        db = MagicMock()
        db.conn = MagicMock()

        with (
            patch("src.main.query_recent_signals", return_value=[sig]),
            patch("src.main.update_signal_outcome"),
        ):
            results = _evaluate_signal_outcomes(db, {})

        assert results == {"winners": [], "losers": [], "open": [], "no_data": []}
