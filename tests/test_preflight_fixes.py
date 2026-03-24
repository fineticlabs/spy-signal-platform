"""Tests for preflight check bug fixes (2026-03-24).

Bug 1: Websocket connectivity check passed string 'iex' instead of DataFeed.IEX enum.
Bug 2: auto_start.sh health check used SIP feed instead of IEX (403 on free plan).
"""

from __future__ import annotations


class TestWebsocketConnectivityCheckUsesEnum:
    """The websocket check must pass DataFeed.IEX (enum), not the string 'iex'."""

    def test_preflight_passes_datafeed_enum(self) -> None:
        """StockDataStream must receive DataFeed.IEX, not a bare string.

        The CLAUDE.md gotcha states: 'DataFeed enum required:
        StockDataStream(feed=DataFeed.IEX) — never pass string "iex"'.
        """

        # Read the source and verify it imports DataFeed and uses the enum
        from pathlib import Path

        preflight_path = Path(__file__).parent.parent / "scripts" / "preflight_check.py"
        source = preflight_path.read_text()

        # Must import the enum
        assert "from alpaca.data.enums import DataFeed" in source

        # Must use DataFeed.IEX (enum), not the string "iex"
        # Find the StockDataStream instantiation in _check_websocket_connectivity
        ws_section_start = source.index("def _check_websocket_connectivity")
        ws_section = source[ws_section_start:]
        # Find the next top-level def to bound the section
        next_def = ws_section.index("\ndef ", 1)
        ws_section = ws_section[:next_def]

        assert (
            "feed=DataFeed.IEX" in ws_section
        ), "StockDataStream should use feed=DataFeed.IEX, not a string"
        # Ensure the old string form is NOT used
        assert 'feed="iex"' not in ws_section, 'StockDataStream must not use feed="iex" (string)'

    def test_datafeed_enum_has_value_attribute(self) -> None:
        """DataFeed.IEX must have .value — this is what the SDK calls internally."""
        from alpaca.data.enums import DataFeed

        assert hasattr(DataFeed.IEX, "value")
        assert DataFeed.IEX.value == "iex"

    def test_string_lacks_value_attribute(self) -> None:
        """A bare string 'iex' does NOT have .value — this was the crash."""
        assert not hasattr("iex", "value")


class TestAutoStartUsesIEXFeed:
    """auto_start.sh health check must use IEX feed, not SIP (403 on free plan)."""

    def test_auto_start_imports_datafeed(self) -> None:
        """The inline Python in auto_start.sh must import DataFeed."""
        from pathlib import Path

        script_path = Path(__file__).parent.parent / "scripts" / "auto_start.sh"
        source = script_path.read_text()

        assert "from alpaca.data.enums import DataFeed" in source

    def test_auto_start_uses_iex_feed(self) -> None:
        """StockBarsRequest must include feed=DataFeed.IEX."""
        from pathlib import Path

        script_path = Path(__file__).parent.parent / "scripts" / "auto_start.sh"
        source = script_path.read_text()

        assert (
            "feed=DataFeed.IEX" in source
        ), "StockBarsRequest must specify feed=DataFeed.IEX to avoid SIP 403"

    def test_auto_start_bars_request_has_feed(self) -> None:
        """The StockBarsRequest line must include an explicit feed parameter."""
        from pathlib import Path

        script_path = Path(__file__).parent.parent / "scripts" / "auto_start.sh"
        lines = script_path.read_text().splitlines()

        bars_request_lines = [ln for ln in lines if "StockBarsRequest" in ln]
        assert bars_request_lines, "No StockBarsRequest found in auto_start.sh"

        # The line containing the request constructor OR a nearby line must have feed=
        # (since it's all one Python -c string, check the full inline script)
        inline_block = "\n".join(lines)
        # Extract the python -c block
        start = inline_block.index("StockBarsRequest")
        # Find the closing of the request (next line with c.get_stock_bars)
        end = inline_block.index("get_stock_bars", start)
        block = inline_block[start:end]
        assert "feed=" in block, "StockBarsRequest must include feed=DataFeed.IEX to avoid SIP 403"
