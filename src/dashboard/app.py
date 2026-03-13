"""Streamlit monitoring dashboard for the SPY Signal Platform.

Polls the FastAPI backend (http://127.0.0.1:8000) every 5 seconds and
renders live trading state: indicators, levels, signals, and risk status.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import contextlib
import time
from datetime import UTC, datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import requests
import streamlit as st

_API_BASE = "http://127.0.0.1:8000"
_ET = ZoneInfo("America/New_York")
_REFRESH_S = 5

_LUNCH_START = dt_time(11, 30)
_LUNCH_END = dt_time(13, 30)
_CUTOFF = dt_time(15, 45)

# Palette
_GREEN = "#4CAF50"
_RED = "#F44336"
_YELLOW = "#FFC107"
_BLUE = "#2196F3"
_GREY = "#9E9E9E"

_REGIME_COLOR: dict[str, str] = {
    "TRENDING_UP": _GREEN,
    "TRENDING_DOWN": _RED,
    "RANGING": _YELLOW,
    "CHOPPY": _GREY,
}
_CONFIDENCE_COLOR: dict[int, str] = {
    1: "#9E9E9E",
    2: "#FF9800",
    3: "#FFC107",
    4: "#8BC34A",
    5: "#4CAF50",
}

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SPY Signal Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API helpers ───────────────────────────────────────────────────────────────


def _get_json(path: str) -> dict | list | None:
    """GET from FastAPI; return None on any network/HTTP error."""
    try:
        resp = requests.get(f"{_API_BASE}{path}", timeout=2)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
    except requests.RequestException:
        return None


def _fetch_all() -> tuple[dict, dict, list, list]:
    """Return (health, state, signals, trades) with empty-safe fallbacks."""
    raw_health = _get_json("/health")
    raw_state = _get_json("/state")
    raw_signals = _get_json("/signals?limit=20")
    raw_trades = _get_json("/trades?days=1&limit=100")

    health: dict = raw_health if isinstance(raw_health, dict) else {}
    state: dict = raw_state if isinstance(raw_state, dict) else {}
    signals: list = raw_signals if isinstance(raw_signals, list) else []
    trades: list = raw_trades if isinstance(raw_trades, list) else []
    return health, state, signals, trades


# ── time helpers ──────────────────────────────────────────────────────────────


def _now_et() -> datetime:
    return datetime.now(_ET)


def _is_lunch() -> bool:
    t = _now_et().time()
    return _LUNCH_START <= t < _LUNCH_END


def _is_after_cutoff() -> bool:
    return _now_et().time() >= _CUTOFF


# ── formatting helpers ────────────────────────────────────────────────────────


def _fmt(val: object, fmt: str = ".2f", prefix: str = "", suffix: str = "") -> str:
    if val is None:
        return "--"
    try:
        return f"{prefix}{float(val):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_price(val: object) -> str:
    return _fmt(val, ".2f", "$")


def _pnl_color(pnl: float) -> str:
    if pnl > 0:
        return _GREEN
    if pnl < 0:
        return _RED
    return _GREY


# ── Plotly figures ────────────────────────────────────────────────────────────


def _rsi_gauge(rsi: float | None) -> go.Figure:
    """RSI gauge with overbought/oversold color zones."""
    value = rsi if rsi is not None else 50.0
    bar_color = _RED if value < 30 or value > 70 else _GREEN

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "" if rsi is not None else " (no data)"},
            title={"text": "RSI(14)", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "rgba(244,67,54,0.25)"},
                    {"range": [30, 70], "color": "rgba(30,30,46,0.0)"},
                    {"range": [70, 100], "color": "rgba(244,67,54,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=180,
        margin={"t": 40, "b": 0, "l": 10, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def _equity_curve_fig(trades: list) -> go.Figure:
    """Cumulative P&L chart for today's trades."""
    fig = go.Figure()

    if trades:
        pnls = [float(t.get("pnl", 0)) for t in trades]
        cumulative: list[float] = []
        running = 0.0
        for p in pnls:
            running += p
            cumulative.append(running)

        positive = [v if v >= 0 else None for v in cumulative]
        negative = [v if v < 0 else None for v in cumulative]

        fig.add_trace(
            go.Scatter(
                y=positive,
                mode="lines",
                line={"color": _GREEN, "width": 2},
                fill="tozeroy",
                fillcolor="rgba(76,175,80,0.15)",
                name="Profit",
                connectgaps=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                y=negative,
                mode="lines",
                line={"color": _RED, "width": 2},
                fill="tozeroy",
                fillcolor="rgba(244,67,54,0.15)",
                name="Loss",
                connectgaps=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                y=cumulative,
                mode="markers",
                marker={"color": [_GREEN if v >= 0 else _RED for v in cumulative], "size": 7},
                name="Trades",
                showlegend=False,
            )
        )

    fig.update_layout(
        title={"text": "Today's Equity Curve", "font": {"size": 13}},
        yaxis_title="P&L ($)",
        xaxis_title="Trade #",
        height=200,
        margin={"t": 35, "b": 10, "l": 45, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,46,0.3)",
        showlegend=False,
        font={"color": "white"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color=_GREY, line_width=1)
    return fig


# ── NO TRADE ZONE banner ──────────────────────────────────────────────────────


def _render_no_trade_banner(risk: dict) -> None:
    """Show a prominent warning banner whenever trading should be paused."""
    reasons: list[str] = []

    if _is_lunch():
        reasons.append("Lunch chop zone (11:30-13:30 ET) — avoid new entries")
    if risk.get("tilted"):
        reasons.append("3 consecutive losses — done for the day")
    if risk.get("cooled_down"):
        reasons.append("Cooldown active — 2 consecutive losses, wait 15 min")
    if _is_after_cutoff():
        reasons.append("After 15:45 ET cutoff — no new entries allowed")

    try:
        daily_pnl = float(risk.get("daily_pnl") or 0)
        if daily_pnl <= -1500:
            reasons.append(f"Daily loss limit hit (${daily_pnl:.2f})")
    except (TypeError, ValueError):
        pass

    if reasons:
        msg = "**\U0001f6ab NO TRADE ZONE**\n\n" + "\n".join(f"- {r}" for r in reasons)
        st.error(msg)


# ── header ────────────────────────────────────────────────────────────────────


def _render_header(levels: dict, regime: dict, health: dict) -> None:
    last_price = levels.get("last_price")
    pdclose = levels.get("prev_day_close")
    vix = regime.get("vix")
    regime_name: str = str(regime.get("name") or "UNKNOWN")
    regime_color = _REGIME_COLOR.get(regime_name, _GREY)
    tradeable: bool = bool(regime.get("tradeable", False))

    daily_change: float | None = None
    daily_change_pct: float | None = None
    if last_price is not None and pdclose is not None:
        try:
            lp, pc = float(last_price), float(pdclose)
            daily_change = lp - pc
            daily_change_pct = (daily_change / pc * 100) if pc > 0 else None
        except (TypeError, ValueError):
            pass

    delta_str: str | None = None
    if daily_change is not None and daily_change_pct is not None:
        delta_str = f"{daily_change:+.2f} ({daily_change_pct:+.2f}%)"

    col1, col2, col3, col4, col5 = st.columns([2, 2, 1.5, 2, 2])

    with col1:
        st.metric(
            label="SPY",
            value=_fmt_price(last_price),
            delta=delta_str,
        )

    with col2:
        vix_f: float | None = None
        with contextlib.suppress(TypeError, ValueError):
            vix_f = float(vix) if vix is not None else None
        vix_label = "VIX"
        if vix_f is not None:
            if vix_f < 15:
                vix_label = "VIX (Low Vol)"
            elif vix_f < 25:
                vix_label = "VIX (Normal)"
            else:
                vix_label = "VIX (HIGH \u26a0)"
        st.metric(label=vix_label, value=_fmt(vix, ".1f"))

    with col3:
        st.metric(
            label="Tradeable",
            value="\u2705 YES" if tradeable else "\u274c NO",
        )

    with col4:
        badge_html = (
            f'<div style="background:{regime_color};color:white;padding:8px 16px;'
            f"border-radius:6px;text-align:center;font-weight:bold;font-size:14px;"
            f'margin-top:4px;">{regime_name}</div>'
        )
        st.markdown("**Regime**")
        st.markdown(badge_html, unsafe_allow_html=True)

    with col5:
        api_ok = health.get("status") == "ok"
        status_str = "\U0001f7e2 Online" if api_ok else "\U0001f534 Offline"
        now_et = _now_et()
        st.markdown(f"**Backend:** {status_str}")
        st.markdown(f"**ET:** `{now_et.strftime('%H:%M:%S')}`")
        is_mkt = dt_time(9, 30) <= now_et.time() < dt_time(16, 0)
        st.markdown("**Market:** " + ("\U0001f7e2 Open" if is_mkt else "\u26ab Closed"))


# ── Row 1: Key Levels ─────────────────────────────────────────────────────────


def _render_levels(levels: dict) -> None:
    st.subheader("Key Levels")

    last_price = levels.get("last_price")
    lp: float | None = None
    with contextlib.suppress(TypeError, ValueError):
        lp = float(last_price) if last_price is not None else None

    rows: list[tuple[str, object]] = [
        ("VWAP", levels.get("vwap")),
        ("VWAP +1\u03c3", levels.get("vwap_upper_1")),
        ("VWAP -1\u03c3", levels.get("vwap_lower_1")),
        ("VWAP +2\u03c3", levels.get("vwap_upper_2")),
        ("VWAP -2\u03c3", levels.get("vwap_lower_2")),
        ("ORB High (5m)", levels.get("orb_high")),
        ("ORB Low (5m)", levels.get("orb_low")),
        ("ORB High (15m)", levels.get("orb15_high")),
        ("ORB Low (15m)", levels.get("orb15_low")),
        ("Prev Day High", levels.get("prev_day_high")),
        ("Prev Day Low", levels.get("prev_day_low")),
        ("Prev Day Close", levels.get("prev_day_close")),
        ("High of Day", levels.get("high_of_day")),
        ("Low of Day", levels.get("low_of_day")),
        ("Premarket High", levels.get("premarket_high")),
        ("Premarket Low", levels.get("premarket_low")),
    ]

    cols = st.columns(4)
    for i, (name, val) in enumerate(rows):
        with cols[i % 4]:
            if val is not None and lp is not None:
                try:
                    fv = float(val)
                    diff = lp - fv
                    pct = diff / fv * 100 if fv != 0 else 0.0
                    st.metric(
                        label=name,
                        value=f"${fv:.2f}",
                        delta=f"{diff:+.2f} ({pct:+.1f}%)",
                    )
                except (TypeError, ValueError):
                    st.metric(label=name, value=_fmt_price(val))
            else:
                st.metric(label=name, value=_fmt_price(val))


# ── Row 2: Indicators ─────────────────────────────────────────────────────────


def _render_indicators(indicators: dict, regime: dict) -> None:
    st.subheader("Indicators")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

    with col1:
        rsi = indicators.get("rsi")
        rsi_f: float | None = None
        with contextlib.suppress(TypeError, ValueError):
            rsi_f = float(rsi) if rsi is not None else None
        st.plotly_chart(_rsi_gauge(rsi_f), use_container_width=True)

    with col2:
        st.markdown("**MACD**")
        st.metric("Line", _fmt(indicators.get("macd"), ".3f"))
        st.metric("Signal", _fmt(indicators.get("macd_signal"), ".3f"))

        hist = indicators.get("macd_histogram")
        if hist is not None:
            try:
                hist_f = float(hist)
                color = _GREEN if hist_f >= 0 else _RED
                st.markdown(
                    f"Histogram: <span style='color:{color};font-weight:bold;'>"
                    f"{hist_f:+.3f}</span>",
                    unsafe_allow_html=True,
                )
            except (TypeError, ValueError):
                st.markdown("Histogram: --")
        else:
            st.markdown("Histogram: --")

        st.markdown("")
        st.metric("ATR(14)", _fmt(indicators.get("atr"), ".3f", "$"))

    with col3:
        st.markdown("**Regime Filters**")

        adx = regime.get("adx")
        adx_f: float | None = None
        with contextlib.suppress(TypeError, ValueError):
            adx_f = float(adx) if adx is not None else None
        adx_label = "ADX"
        if adx_f is not None:
            if adx_f < 15:
                adx_label = "ADX (Choppy)"
            elif adx_f < 25:
                adx_label = "ADX (Weak)"
            else:
                adx_label = "ADX (Trending)"
        st.metric(adx_label, _fmt(adx, ".1f"))

        st.markdown("")
        st.markdown("**Bollinger Bands**")
        st.metric("BB Upper", _fmt_price(indicators.get("bb_upper")))
        st.metric("BB Middle", _fmt_price(indicators.get("bb_middle")))
        st.metric("BB Lower", _fmt_price(indicators.get("bb_lower")))

    with col4:
        st.markdown("**EMA Alignment**")
        e9 = indicators.get("ema9")
        e20 = indicators.get("ema20")
        e50 = indicators.get("ema50")

        if e9 is not None and e20 is not None and e50 is not None:
            try:
                f9, f20, f50 = float(e9), float(e20), float(e50)
                if f9 > f20 > f50:
                    align_label = "\U0001f7e2 BULLISH  9 > 20 > 50"
                    align_color = _GREEN
                elif f9 < f20 < f50:
                    align_label = "\U0001f534 BEARISH  9 < 20 < 50"
                    align_color = _RED
                else:
                    align_label = "\U0001f7e1 MIXED / NEUTRAL"
                    align_color = _YELLOW
                st.markdown(
                    f'<div style="background:{align_color};color:white;padding:8px 12px;'
                    f'border-radius:6px;font-weight:bold;margin-bottom:8px;">'
                    f"{align_label}</div>",
                    unsafe_allow_html=True,
                )
            except (TypeError, ValueError):
                st.markdown("--")
        else:
            st.markdown(
                '<div style="color:#9E9E9E;padding:8px;">Warming up...</div>',
                unsafe_allow_html=True,
            )

        st.metric("EMA 9", _fmt_price(e9))
        st.metric("EMA 20", _fmt_price(e20))
        st.metric("EMA 50", _fmt_price(e50))


# ── Row 3: Active Signals ─────────────────────────────────────────────────────


def _render_signals(signals: list) -> None:
    st.subheader("Recent Signals")
    if not signals:
        st.info("No signals in the last 24 hours.")
        return

    for sig in signals[:5]:
        direction: str = str(sig.get("direction", ""))
        strategy: str = str(sig.get("strategy_name", ""))
        confidence: int = int(sig.get("confidence", sig.get("confidence_score", 3)) or 3)
        approved: bool = bool(sig.get("approved", False))
        ts: str = str(sig.get("timestamp", ""))[:19].replace("T", " ")
        reason: str = str(sig.get("reason", ""))

        entry = sig.get("entry_price")
        stop = sig.get("stop_price")
        target = sig.get("target_price")
        rr = sig.get("risk_reward", sig.get("risk_reward_ratio"))

        dir_color = _GREEN if direction == "LONG" else _RED
        conf_color = _CONFIDENCE_COLOR.get(confidence, _GREY)
        status_badge = "\u2705 APPROVED" if approved else "\u274c REJECTED"
        status_color = _GREEN if approved else _RED

        card_html = (
            '<div style="border:1px solid #333;border-radius:8px;padding:12px;'
            "margin-bottom:8px;background:rgba(30,30,46,0.6);\">"
            '<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:{dir_color};font-size:18px;font-weight:bold;">'
            f"{direction} \u2014 {strategy}</span>"
            f'<span style="background:{conf_color};color:black;padding:2px 10px;'
            f'border-radius:12px;font-weight:bold;">\u2605 {confidence}/5</span>'
            f'<span style="color:{status_color};font-weight:bold;">{status_badge}</span>'
            "</div>"
            '<div style="margin-top:6px;font-size:13px;color:#CCC;">'
            f"Entry: <b>{_fmt_price(entry)}</b> &nbsp;|\u00a0"
            f"Stop: <b>{_fmt_price(stop)}</b> &nbsp;|\u00a0"
            f"Target: <b>{_fmt_price(target)}</b> &nbsp;|\u00a0"
            f"R:R <b>{_fmt(rr, '.2f')}</b>"
            "</div>"
            f'<div style="margin-top:4px;font-size:12px;color:#AAA;">{reason}</div>'
            f'<div style="margin-top:2px;font-size:11px;color:#666;">{ts} UTC</div>'
            "</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)


# ── Row 4: Alerts Log ─────────────────────────────────────────────────────────


def _render_alerts_log(signals: list) -> None:
    st.subheader("Alerts Log (Last 20)")
    if not signals:
        st.info("No alerts yet.")
        return

    import pandas as pd

    rows = []
    for s in signals[:20]:
        approved = bool(s.get("approved", False))
        rows.append(
            {
                "Time (UTC)": str(s.get("timestamp", ""))[:19].replace("T", " "),
                "Strategy": s.get("strategy_name", ""),
                "Dir": s.get("direction", ""),
                "Entry": _fmt_price(s.get("entry_price")),
                "Stop": _fmt_price(s.get("stop_price")),
                "Target": _fmt_price(s.get("target_price")),
                "R:R": _fmt(s.get("risk_reward", s.get("risk_reward_ratio")), ".2f"),
                "Conf": s.get("confidence", s.get("confidence_score", "")),
                "Status": "\u2705 OK" if approved else "\u274c Rej",
                "Reason": str(s.get("reason", ""))[:60],
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=min(300, 38 + len(rows) * 35))


# ── Row 5: Daily P&L & Risk ───────────────────────────────────────────────────


def _render_risk_panel(risk: dict, trades: list) -> None:
    st.subheader("Daily P&L & Risk Status")

    daily_pnl: float = 0.0
    with contextlib.suppress(TypeError, ValueError):
        daily_pnl = float(risk.get("daily_pnl") or 0)

    daily_trades: int = int(risk.get("daily_trades", 0) or 0)
    consec_losses: int = int(risk.get("consecutive_losses", 0) or 0)
    cooled_down: bool = bool(risk.get("cooled_down", False))
    tilted: bool = bool(risk.get("tilted", False))

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        pnl_color = _pnl_color(daily_pnl)
        st.markdown("**Daily P&L**")
        st.markdown(
            f'<div style="font-size:26px;font-weight:bold;color:{pnl_color};">'
            f"${daily_pnl:+.2f}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        trades_color = _RED if daily_trades >= 5 else _GREEN
        st.markdown("**Trades Today**")
        st.markdown(
            f'<div style="font-size:22px;font-weight:bold;color:{trades_color};">'
            f"{daily_trades}/5</div>",
            unsafe_allow_html=True,
        )

    with col3:
        loss_color = _RED if consec_losses >= 2 else (_YELLOW if consec_losses == 1 else _GREEN)
        st.markdown("**Consec. Losses**")
        st.markdown(
            f'<div style="font-size:22px;font-weight:bold;color:{loss_color};">'
            f"{consec_losses}/3</div>",
            unsafe_allow_html=True,
        )

    with col4:
        cd_color = _RED if cooled_down else _GREEN
        cd_label = "\U0001f534 ACTIVE (15 min)" if cooled_down else "\U0001f7e2 Clear"
        st.markdown("**Cooldown**")
        st.markdown(
            f'<div style="font-size:15px;font-weight:bold;color:{cd_color};">' f"{cd_label}</div>",
            unsafe_allow_html=True,
        )

    with col5:
        tilt_color = _RED if tilted else _GREEN
        tilt_label = "\U0001f6ab TILTED \u2014 STOP" if tilted else "\U0001f7e2 OK"
        st.markdown("**Tilt Status**")
        st.markdown(
            f'<div style="font-size:15px;font-weight:bold;color:{tilt_color};">'
            f"{tilt_label}</div>",
            unsafe_allow_html=True,
        )

    st.plotly_chart(_equity_curve_fig(trades), use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _render_sidebar(health: dict, state: dict, trades: list) -> None:
    with st.sidebar:
        st.title("\U0001f4c8 System Health")

        api_ok = health.get("status") == "ok"
        conn_str = "\U0001f7e2 Connected" if api_ok else "\U0001f534 Disconnected"
        st.markdown(f"**API:** {conn_str}")
        st.markdown(f"**Host:** `{_API_BASE}`")

        # Data freshness from health timestamp
        if health.get("timestamp"):
            try:
                ts_str = str(health["timestamp"])
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age_s = (datetime.now(UTC) - ts).total_seconds()
                if age_s < 10:
                    freshness = "\U0001f7e2 Fresh"
                elif age_s < 60:
                    freshness = "\U0001f7e1 Stale"
                else:
                    freshness = "\U0001f534 Old"
                st.markdown(f"**Data:** {freshness} ({age_s:.0f}s ago)")
            except (ValueError, AttributeError):
                pass

        # Last price as proxy for last bar time
        levels: dict = state.get("levels") or {}
        last_price = levels.get("last_price")
        if last_price is not None:
            st.markdown(f"**Last Price:** ${float(last_price):.2f}")

        st.divider()

        # Market time
        now_et = _now_et()
        st.markdown(f"**ET:** `{now_et.strftime('%Y-%m-%d %H:%M:%S')}`")
        is_mkt = dt_time(9, 30) <= now_et.time() < dt_time(16, 0)
        st.markdown("**Market:** " + ("\U0001f7e2 Open" if is_mkt else "\u26ab Closed"))
        if _is_lunch():
            st.warning("Lunch chop zone")
        if _is_after_cutoff():
            st.warning("After 15:45 cutoff")

        st.divider()

        # Regime summary
        regime: dict = state.get("regime") or {}
        regime_name = str(regime.get("name") or "--")
        regime_color = _REGIME_COLOR.get(regime_name, _GREY)
        st.markdown(
            f'<span style="background:{regime_color};color:white;padding:3px 10px;'
            f'border-radius:10px;font-weight:bold;">{regime_name}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**VIX:** {_fmt(regime.get('vix'), '.1f')}")
        st.markdown(f"**ADX:** {_fmt(regime.get('adx'), '.1f')}")
        tradeable = bool(regime.get("tradeable", False))
        st.markdown("**Trading:** " + ("\u2705 Active" if tradeable else "\u274c Paused"))

        st.divider()

        # Today's trade count
        risk: dict = state.get("risk") or {}
        daily_pnl: float = 0.0
        with contextlib.suppress(TypeError, ValueError):
            daily_pnl = float(risk.get("daily_pnl") or 0)
        pnl_color = _pnl_color(daily_pnl)
        st.markdown(
            f"**Daily P&L:** <span style='color:{pnl_color};font-weight:bold;'>"
            f"${daily_pnl:+.2f}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Trades:** {len(trades)} today")

        st.divider()
        st.caption(f"\U0001f504 Auto-refresh every {_REFRESH_S}s")
        if st.button("\U0001f504 Refresh Now"):
            st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Fetch data, render all sections, then sleep and rerun."""
    health, state, signals, trades = _fetch_all()

    indicators: dict = state.get("indicators") or {}
    levels: dict = state.get("levels") or {}
    regime: dict = state.get("regime") or {}
    risk: dict = state.get("risk") or {}

    _render_sidebar(health, state, trades)

    st.title("\U0001f4c8 SPY Signal Platform")

    _render_no_trade_banner(risk)

    _render_header(levels, regime, health)

    st.divider()
    _render_levels(levels)

    st.divider()
    _render_indicators(indicators, regime)

    st.divider()
    _render_signals(signals)

    st.divider()
    _render_alerts_log(signals)

    st.divider()
    _render_risk_panel(risk, trades)

    # Auto-refresh
    time.sleep(_REFRESH_S)
    st.rerun()


if __name__ == "__main__":
    main()
