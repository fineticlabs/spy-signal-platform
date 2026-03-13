"""Confidence-filtered backtest: compare no-ML vs >60% vs >70% confidence.

Purged walk-forward evaluation
-------------------------------
For each trade in window N (walk-forward index), the ML model is trained
only on features from windows 0..N-1, with a 5-calendar-day purge gap
between the last training row and the first test row.  This exactly
mirrors how the model would be used in production.

We require a minimum of MIN_TRAIN_TRADES in the training set before
scoring any test window.  Windows below this threshold are skipped for the
ML comparison (but counted in the baseline).

Three result versions
---------------------
A) Baseline   : all backtest trades, no ML filter (matches run_backtest.py)
B) ML > 60%   : only trades where model confidence > 0.60
C) ML > 70%   : only trades where model confidence > 0.70
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog

from src.backtest.metrics import compute_metrics, save_equity_curve
from src.backtest.ml_features import FEATURE_COLS, extract_all_features
from src.backtest.ml_scorer import _CATEGORICAL_FEATURES, _PURGE_DAYS
from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)

_MIN_TRAIN_TRADES = 50  # minimum trades before ML scoring begins
_CONF_60 = 0.60
_CONF_70 = 0.70

# ── purged walk-forward scoring ───────────────────────────────────────────────


def _score_with_purged_cv(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'confidence' column via purged walk-forward model training.

    For each unique window_idx W (ascending), trains a LightGBM on all
    rows from window 0..W-1 with entry_time < (W's first entry - 5 days),
    then predicts confidence for all rows in window W.

    Windows with insufficient training data receive confidence = NaN.

    Args:
        features_df: Output of extract_all_features().  Must be sorted by
                     entry_time.

    Returns:
        Input DataFrame with an added 'confidence' column (float, 0-1 or NaN).
    """
    df = features_df.copy()
    df["confidence"] = np.nan

    windows = sorted(df["window_idx"].unique())
    logger.info("purged_cv_scoring", total_windows=len(windows))

    for w in windows:
        test_mask = df["window_idx"] == w
        test_df = df[test_mask]
        if test_df.empty:
            continue

        # Purge cutoff: 5 calendar days before first test entry
        cutoff = test_df["entry_time"].min() - pd.Timedelta(days=_PURGE_DAYS)
        train_mask = (df["window_idx"] < w) & (df["entry_time"] < cutoff)
        train_df = df[train_mask].dropna(subset=[*FEATURE_COLS, "label"])

        if len(train_df) < _MIN_TRAIN_TRADES:
            # Not enough history yet — skip (NaN confidence = excluded from ML sets)
            continue

        x_train = train_df[FEATURE_COLS].copy()
        x_train["ticker_encoded"] = x_train["ticker_encoded"].astype(int)
        y_train = train_df["label"].astype(int)

        x_test = test_df[FEATURE_COLS].dropna().copy()
        if x_test.empty:
            continue
        x_test["ticker_encoded"] = x_test["ticker_encoded"].astype(int)

        try:
            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
                n_jobs=1,
            )
            model.fit(
                x_train,
                y_train,
                categorical_feature=_CATEGORICAL_FEATURES,
            )
            proba = model.predict_proba(x_test)[:, 1]
            df.loc[x_test.index, "confidence"] = proba

        except Exception as exc:
            logger.warning("window_scoring_failed", window=w, error=str(exc))

    scored = int((~df["confidence"].isna()).sum())
    logger.info("purged_cv_done", scored=scored, total=len(df))
    return df


# ── metrics helper ────────────────────────────────────────────────────────────


def _metrics_from_pnl_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Compute metrics dict from a features DataFrame (has 'pnl' column).

    Constructs a synthetic _trades-compatible DataFrame so compute_metrics
    can be reused directly.

    Args:
        df: Filtered features DataFrame with 'pnl' and 'entry_time' columns.

    Returns:
        Metrics dict from compute_metrics().
    """
    if df.empty:
        return compute_metrics(pd.DataFrame())

    # Build a minimal _trades-like DataFrame
    trades = pd.DataFrame(
        {
            "PnL": df["pnl"].to_numpy(dtype=float),
            "EntryTime": df["entry_time"].to_numpy(),
        }
    )
    equity = pd.Series(trades["PnL"].cumsum().to_numpy())
    return compute_metrics(trades, equity_curve=equity)


# ── summary printer ───────────────────────────────────────────────────────────


def _print_ml_summary(label: str, df: pd.DataFrame) -> None:
    """Print a compact summary line + per-year breakdown for one filter set.

    Args:
        label: Version name (e.g. 'Baseline', 'ML >60%').
        df:    Filtered features DataFrame.
    """
    if df.empty:
        print(f"\n  {label}: 0 trades")
        return

    m = _metrics_from_pnl_frame(df)
    pf = m.get("profit_factor", 0.0)
    pf_str = f"{float(pf):.3f}" if float(pf) != float("inf") else "inf"

    net = float(m.get("net_profit", 0.0))
    exp = float(m.get("expectancy", 0.0))
    wr = float(m.get("win_rate", 0.0)) * 100
    tot = m.get("total_trades", 0)
    sharpe = float(m.get("sharpe_ratio", 0.0))

    sep = "-" * 60
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(sep)
    print(f"  Trades:       {tot}")
    print(f"  Win rate:     {wr:.1f}%")
    print(f"  Profit factor:{pf_str:>8}")
    print(f"  Expectancy:   ${exp:.2f}/trade")
    print(f"  Net profit:   ${net:.2f}")
    print(f"  Sharpe ratio: {sharpe:.3f}")

    # Per-year breakdown
    year: dict[str, Any] = m.get("year_breakdown", {})
    if year:
        print(sep)
        print("  Per-year:")
        for yr, vals in year.items():
            sign = "+" if float(vals["total_pnl"]) >= 0 else ""
            print(
                f"  {yr}: {int(vals['trades']):>4} trades, "
                f"WR={float(vals['win_rate'])*100:.0f}%, "
                f"PnL={sign}${float(vals['total_pnl']):.2f}"
            )
    print("=" * 60)


# ── main entry point ──────────────────────────────────────────────────────────


def run_ml_backtest(
    symbols: list[str],
    features_csv: Path | None = None,
    is_days: int = 60,
    oos_days: int = 20,
    out_dir: Path = Path("docs"),
) -> None:
    """Run the full confidence-filtered backtest comparison.

    Extracts features (or loads cached CSV), scores with purged walk-forward
    CV, then prints three comparison tables:
        A) Baseline (no ML filter)
        B) ML confidence > 60%
        C) ML confidence > 70%

    Args:
        symbols:      Ticker symbols to include.
        features_csv: Optional path to a pre-cached features CSV.
        is_days:      In-sample window size for walk-forward splits.
        oos_days:     Out-of-sample window size.
        out_dir:      Directory for equity curve PNG output.
    """
    # ── load or extract features ──────────────────────────────────────────
    if features_csv and features_csv.exists():
        logger.info("loading_features_csv", path=str(features_csv))
        all_features = pd.read_csv(features_csv, parse_dates=["entry_time"])
        all_features["entry_time"] = pd.to_datetime(all_features["entry_time"], utc=True)
        print(f"  Loaded {len(all_features):,} trades from {features_csv}")
    else:
        print("\nExtracting features from database...")
        db = BarDatabase()
        db.connect()
        all_features = extract_all_features(db, symbols, is_days, oos_days)
        db.close()
        if all_features.empty:
            print("No features extracted. Run backfill_data.py first.")
            return

        if features_csv:
            features_csv.parent.mkdir(parents=True, exist_ok=True)
            all_features.to_csv(features_csv, index=False)
            print(f"  Features cached → {features_csv}")

    # ── purged walk-forward scoring ───────────────────────────────────────
    print(f"\nScoring {len(all_features):,} trades with purged walk-forward CV...")
    scored_df = _score_with_purged_cv(all_features)

    n_scored = int((~scored_df["confidence"].isna()).sum())
    n_unscored = int(scored_df["confidence"].isna().sum())
    print(f"  Scored: {n_scored} | Unscored (early windows): {n_unscored}")

    # ── three versions ────────────────────────────────────────────────────
    # A: Baseline — all trades (no ML filter)
    _print_ml_summary("A) Baseline — all trades (no ML filter)", scored_df)

    # B: ML > 60% — only from scored windows
    scored_only = scored_df[~scored_df["confidence"].isna()]
    ml60_df = scored_only[scored_only["confidence"] > _CONF_60]
    _print_ml_summary(f"B) ML confidence > {_CONF_60:.0%}", ml60_df)

    # C: ML > 70%
    ml70_df = scored_only[scored_only["confidence"] > _CONF_70]
    _print_ml_summary(f"C) ML confidence > {_CONF_70:.0%}", ml70_df)

    # ── comparison table ──────────────────────────────────────────────────
    print("\n\n  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  Three-Way Comparison (scored windows only)                         │")
    print("  ├──────────────┬────────┬────────┬────────┬──────────┬─────────────  │")
    print("  │ Version      │ Trades │ WinRate│   PF   │ $/trade  │  Net PnL      │")
    print("  ├──────────────┼────────┼────────┼────────┼──────────┼─────────────  │")

    def _row(label: str, df: pd.DataFrame) -> None:
        if df.empty:
            print(f"  │ {label:<12} │      0 │    n/a │    n/a │      n/a │          n/a  │")
            return
        m = _metrics_from_pnl_frame(df)
        pf = float(m.get("profit_factor", 0))
        pf_s = f"{pf:.3f}" if pf != float("inf") else "  inf"
        print(
            f"  │ {label:<12} │ {int(m['total_trades']):>6} │ "
            f"{float(m['win_rate'])*100:>5.1f}% │ {pf_s:>6} │ "
            f"${float(m['expectancy']):>7.2f} │ ${float(m['net_profit']):>10.2f}  │"
        )

    _row("No ML", scored_only)
    _row(">60%", ml60_df)
    _row(">70%", ml70_df)
    print("  └──────────────┴────────┴────────┴────────┴──────────┴─────────────  ┘")

    # ── equity curves ─────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    best_label = "No ML"
    best_df = scored_only
    if not ml60_df.empty and _metrics_from_pnl_frame(ml60_df).get(
        "profit_factor", 0
    ) > _metrics_from_pnl_frame(scored_only).get("profit_factor", 0):
        best_label = ">60%"
        best_df = ml60_df
    if not ml70_df.empty and _metrics_from_pnl_frame(ml70_df).get(
        "profit_factor", 0
    ) > _metrics_from_pnl_frame(best_df).get("profit_factor", 0):
        best_label = ">70%"
        best_df = ml70_df

    eq = pd.Series(best_df["pnl"].cumsum().to_numpy())
    curve_path = out_dir / "ml_equity_curve.png"
    save_equity_curve(
        eq,
        curve_path,
        title=f"ML-Filtered ORB Equity Curve ({best_label} — {', '.join(symbols)})",
    )
    print(f"\n  Best version: {best_label}")
    print(f"  Equity curve saved → {curve_path}")
