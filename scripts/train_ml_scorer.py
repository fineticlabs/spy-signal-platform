#!/usr/bin/env python
"""CLI: extract features → tune → train LightGBM → save → ML backtest comparison.

Usage
-----
    python scripts/train_ml_scorer.py [--symbols SPY,QQQ,...] [--trials 200]
                                      [--is-days 60] [--oos-days 20]
                                      [--skip-train] [--features-csv PATH]

Steps
-----
1. Extract ORB signal features from the 7-ticker walk-forward backtest.
2. Cache features to docs/ml_features.csv (skipped if --features-csv exists).
3. Tune LightGBM hyperparameters with Optuna (200 trials, purged 5-fold CV).
4. Train final model on ALL features.
5. Save model to models/lgbm_signal_scorer.pkl.
6. Generate SHAP feature importance plot → docs/shap_feature_importance.png.
7. Run three-version ML confidence-filtered backtest and print comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.ml_backtest import run_ml_backtest
from src.backtest.ml_features import TICKER_LIST, extract_all_features
from src.backtest.ml_scorer import _MODEL_PATH, _SHAP_PATH, tune_and_train
from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)

_DEFAULT_FEATURES_CSV = Path("docs/ml_features.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM signal scorer and run confidence-filtered backtest."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(TICKER_LIST),
        help=f"Comma-separated tickers (default: {','.join(TICKER_LIST)})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Number of Optuna trials (default: 200)",
    )
    parser.add_argument(
        "--is-days",
        type=int,
        default=60,
        help="In-sample window days (default: 60)",
    )
    parser.add_argument(
        "--oos-days",
        type=int,
        default=20,
        help="Out-of-sample window days (default: 20)",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=_DEFAULT_FEATURES_CSV,
        help=f"Cached features CSV path (default: {_DEFAULT_FEATURES_CSV})",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; use existing model for backtest only",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs"),
        help="Output directory for plots and CSV (default: docs)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols specified.")
        sys.exit(1)

    print(f"\n{'=' * 65}")
    print("  ML Signal Scorer Training Pipeline")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  IS/OOS  : {args.is_days}/{args.oos_days} days")
    print(f"  Trials  : {args.trials}")
    print(f"{'=' * 65}")

    # ── step 1-2: extract or load features ───────────────────────────────
    features_csv: Path = args.features_csv

    if features_csv.exists():
        import pandas as pd

        print(f"\nLoading cached features from {features_csv}...")
        features_df = pd.read_csv(features_csv, parse_dates=["entry_time"])
        features_df["entry_time"] = pd.to_datetime(features_df["entry_time"], utc=True)
        print(f"  Loaded {len(features_df):,} trade rows")
    else:
        print("\nStep 1/3: Extracting features from walk-forward backtest...")
        db = BarDatabase()
        db.connect()
        features_df = extract_all_features(
            db=db,
            symbols=symbols,
            is_days=args.is_days,
            oos_days=args.oos_days,
        )
        db.close()

        if features_df.empty:
            print("No features extracted. Ensure bars are loaded (backfill_data.py).")
            sys.exit(1)

        features_csv.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_csv, index=False)
        print(f"  Features saved → {features_csv}")

    print(
        f"\n  Total trades: {len(features_df):,}  |  "
        f"Win rate: {features_df['label'].mean():.1%}  |  "
        f"Symbols: {features_df['symbol'].nunique()}"
    )

    # ── step 3-6: tune + train + SHAP ────────────────────────────────────
    if not args.skip_train:
        print(f"\nStep 2/3: Tuning LightGBM ({args.trials} Optuna trials)...")
        print("         (purged 5-fold time-series CV, optimising ROC-AUC)")
        tune_and_train(
            features_df=features_df,
            n_trials=args.trials,
            model_path=_MODEL_PATH,
            shap_path=_SHAP_PATH,
        )
    else:
        print(f"\nSkipping training — using existing model at {_MODEL_PATH}")
        if not _MODEL_PATH.exists():
            print(f"ERROR: No model found at {_MODEL_PATH}")
            sys.exit(1)

    # ── step 7: ML confidence-filtered backtest ───────────────────────────
    print("\nStep 3/3: Running ML confidence-filtered backtest comparison...")
    run_ml_backtest(
        symbols=symbols,
        features_csv=features_csv,
        is_days=args.is_days,
        oos_days=args.oos_days,
        out_dir=args.out_dir,
    )

    print(f"\n{'=' * 65}")
    print("  Done.  Outputs:")
    print(f"    Model   : {_MODEL_PATH}")
    print(f"    SHAP    : {_SHAP_PATH}")
    print(f"    Features: {features_csv}")
    print(f"    Curves  : {args.out_dir}/ml_equity_curve.png")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
