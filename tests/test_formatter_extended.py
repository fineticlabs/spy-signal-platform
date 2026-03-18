"""Extended tests for src.alerts.formatter — covering level snapshots, earnings blackout, and helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.alerts.formatter import (
    _md2,
    _price,
    _stars,
    format_earnings_blackout_alert,
    format_signal_alert,
)
from src.models import (
    Direction,
    LevelSnapshot,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
)


def _make_signal(
    *,
    tags: list[str] | None = None,
    levels_snapshot: LevelSnapshot | None = None,
) -> Signal:
    """Build a Signal with sensible defaults; override tags and levels as needed."""
    return Signal(
        symbol="SPY",
        direction=Direction.LONG,
        strategy_name="ORB",
        entry_price=Decimal("480.00"),
        stop_price=Decimal("478.00"),
        target_price=Decimal("484.00"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=4,
        reason="ORB breakout confirmed",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        vix=Decimal("18.5"),
        adx=Decimal("28.0"),
        timestamp=datetime(2024, 1, 15, 14, 36, tzinfo=UTC),
        tags=tags if tags is not None else [],
        levels_snapshot=levels_snapshot,
    )


def _make_risk_decision(position_size: int = 100) -> RiskDecision:
    return RiskDecision(approved=True, reason="OK", position_size=position_size)


# ── Helper tests ─────────────────────────────────────────────────────────────


class TestMd2:
    """Tests for the _md2 MarkdownV2 escape function."""

    def test_escapes_all_special_characters(self) -> None:
        special = r"\_*[]()~`>#+-=|{}.!"
        result = _md2(special)
        for ch in special:
            assert f"\\{ch}" in result

    def test_plain_text_unchanged(self) -> None:
        assert _md2("hello world") == "hello world"

    def test_mixed_text(self) -> None:
        assert _md2("price=1.50") == "price\\=1\\.50"


class TestPrice:
    """Tests for the _price helper."""

    def test_none_returns_na(self) -> None:
        result = _price(None)
        assert "N/A" in result

    def test_decimal_formatted(self) -> None:
        result = _price(Decimal("480.123"))
        assert "480" in result
        assert "12" in result  # rounded to 2dp


class TestStars:
    """Tests for the _stars confidence display."""

    def test_zero_all_empty(self) -> None:
        result = _stars(0)
        assert result.count("☆") == 5
        assert "⭐" not in result

    def test_five_all_filled(self) -> None:
        result = _stars(5)
        assert result.count("⭐") == 5
        assert "☆" not in result

    def test_three_mixed(self) -> None:
        result = _stars(3)
        assert result.count("⭐") == 3
        assert result.count("☆") == 2

    def test_negative_clamps_to_zero(self) -> None:
        result = _stars(-1)
        assert result.count("☆") == 5

    def test_over_five_clamps(self) -> None:
        result = _stars(8)
        assert result.count("⭐") == 5


# ── format_signal_alert level snapshot lines ─────────────────────────────────


class TestFormatSignalAlertLevels:
    """Cover level snapshot branches: ORB, VWAP, PDH/PDL, RVOL, VP, VIX term, HMM, Kalman."""

    def test_full_level_snapshot_lines_present(self) -> None:
        lvl = LevelSnapshot(
            orb_high=Decimal("481.00"),
            orb_low=Decimal("479.00"),
            vwap=Decimal("480.50"),
            prev_day_high=Decimal("482.00"),
            prev_day_low=Decimal("477.00"),
        )
        sig = _make_signal(levels_snapshot=lvl)
        msg = format_signal_alert(sig, _make_risk_decision())

        assert "ORB:" in msg
        assert "VWAP:" in msg
        assert "PDH/PDL:" in msg

    def test_rvol_line_present(self) -> None:
        lvl = LevelSnapshot(rvol=Decimal("2.35"))
        sig = _make_signal(levels_snapshot=lvl)
        msg = format_signal_alert(sig, _make_risk_decision())

        assert "RVOL:" in msg

    def test_vp_poc_line_present(self) -> None:
        lvl = LevelSnapshot(
            vp_poc=Decimal("480.10"),
            vp_vah=Decimal("481.50"),
            vp_val=Decimal("478.80"),
        )
        sig = _make_signal(levels_snapshot=lvl)
        msg = format_signal_alert(sig, _make_risk_decision())

        assert "VP POC:" in msg
        assert "VA:" in msg

    def test_vix_term_line_present(self) -> None:
        lvl = LevelSnapshot(vix_term_ratio=Decimal("0.82"))
        sig = _make_signal(levels_snapshot=lvl)
        msg = format_signal_alert(sig, _make_risk_decision())

        assert "VIX Term:" in msg
        assert "0\\.82" in msg

    def test_hmm_regime_line_present(self) -> None:
        lvl = LevelSnapshot(hmm_regime="VOLATILE")
        sig = _make_signal(levels_snapshot=lvl)
        msg = format_signal_alert(sig, _make_risk_decision())

        assert "HMM:" in msg
        assert "VOLATILE" in msg

    def test_kalman_stop_mult_line_present(self) -> None:
        lvl = LevelSnapshot(kalman_stop_mult=Decimal("1.20"))
        sig = _make_signal(levels_snapshot=lvl)
        msg = format_signal_alert(sig, _make_risk_decision())

        assert "Stop Mult:" in msg


# ── Tag formatting ───────────────────────────────────────────────────────────


class TestFormatSignalAlertTags:
    """Cover the tag_str empty branch (line 68)."""

    def test_no_tags_field_omitted(self) -> None:
        """Signal with tags=None → no bracket characters in output."""
        sig = _make_signal(tags=None)
        msg = format_signal_alert(sig, _make_risk_decision())
        # The header line should NOT contain brackets around tags
        first_line = msg.split("\n")[0]
        assert "[" not in first_line

    def test_empty_tags_list_no_brackets(self) -> None:
        sig = _make_signal(tags=[])
        msg = format_signal_alert(sig, _make_risk_decision())
        first_line = msg.split("\n")[0]
        assert "[" not in first_line

    def test_tags_present_in_header(self) -> None:
        sig = _make_signal(tags=["ENGULFING", "COMPRESSED"])
        msg = format_signal_alert(sig, _make_risk_decision())
        first_line = msg.split("\n")[0]
        assert "ENGULFING" in first_line
        assert "COMPRESSED" in first_line


# ── Earnings blackout ────────────────────────────────────────────────────────


class TestFormatEarningsBlackoutAlert:
    """Cover format_earnings_blackout_alert (line 134-151)."""

    def test_contains_symbol(self) -> None:
        msg = format_earnings_blackout_alert("AAPL", "2024-01-25")
        assert "AAPL" in msg

    def test_contains_earnings_date(self) -> None:
        msg = format_earnings_blackout_alert("MSFT", "2024-02-01")
        assert "2024\\-02\\-01" in msg

    def test_contains_blackout_keyword(self) -> None:
        msg = format_earnings_blackout_alert("TSLA", "2024-03-10")
        assert "BLACKOUT" in msg

    def test_markdown_structure(self) -> None:
        msg = format_earnings_blackout_alert("GOOG", "2024-04-20")
        assert msg.startswith("\U0001f6ab")  # 🚫
        assert "Resumes day after next" in msg
