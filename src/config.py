from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALPACA_", env_file=".env", extra="ignore")

    api_key: str = Field(..., description="Alpaca API key")
    secret_key: str = Field(..., description="Alpaca secret key")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca base URL (paper or live)",
    )
    feed: str = Field(default="iex", description="Market data feed: 'iex' or 'sip'")

    @field_validator("feed")
    @classmethod
    def validate_feed(cls, v: str) -> str:
        if v not in ("iex", "sip"):
            raise ValueError(f"feed must be 'iex' or 'sip', got {v!r}")
        return v


class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", env_file=".env", extra="ignore")

    bot_token: str = Field(..., description="Telegram bot token from @BotFather")
    chat_id: str = Field(..., description="Telegram chat ID to send alerts to")


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    account_size: Decimal = Field(default=Decimal("50000"), description="Account size in USD")
    risk_per_trade_pct: Decimal = Field(
        default=Decimal("1.0"), description="Max risk per trade as % of account"
    )
    max_daily_loss_pct: Decimal = Field(
        default=Decimal("3.0"), description="Max daily loss as % before stopping"
    )
    max_trades_per_day: int = Field(default=5, description="Max number of trades per day")
    max_concurrent_positions: int = Field(
        default=3, description="Max concurrent open positions (informational)"
    )
    position_scale_factor: float = Field(
        default=0.25,
        description=(
            "Multiplier applied to computed position size (0.25 = 25% of base size). "
            "Shared by both live scanner and backtest engine."
        ),
    )

    @field_validator("position_scale_factor")
    @classmethod
    def validate_scale_factor(cls, v: float) -> float:
        if v <= 0 or v > 1.0:
            raise ValueError(
                f"position_scale_factor must be between 0 (exclusive) and 1.0, got {v}"
            )
        return v

    @field_validator("risk_per_trade_pct", "max_daily_loss_pct")
    @classmethod
    def validate_pct(cls, v: Decimal) -> Decimal:
        if v <= 0 or v > 100:
            raise ValueError(f"percentage must be between 0 and 100, got {v}")
        return v

    @field_validator("max_trades_per_day", "max_concurrent_positions")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"value must be >= 1, got {v}")
        return v


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = Field(default="INFO", description="Log level")
    trading_mode: str = Field(
        default="paper", description="Trading mode: 'live', 'paper', or 'backtest'"
    )
    db_path: str = Field(default="data/spy_signals.db", description="Path to SQLite database")
    symbols: list[str] = Field(
        default=[
            # ── Original 15 ──────────────────────────────────────────────
            "SPY",
            "QQQ",
            "MSFT",
            "AMD",
            "TSLA",
            "AMZN",
            "UBER",
            "SMCI",
            "SHOP",
            "PLTR",
            "NFLX",
            "MSTR",
            "SNOW",
            "ARM",
            "DASH",
            # ── Expansion (added 2026-03-16) ─────────────────────────────
            "PYPL",
            "INTC",
            "MU",
            "HOOD",
            "DKNG",
            "SOXL",  # 3x leveraged semiconductor ETF
            "ROKU",
            "TQQQ",  # 3x leveraged Nasdaq-100 ETF
            "BA",
            "MRVL",
            "META",
        ],
        description="Ticker symbols to trade (comma-separated in .env)",
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return [str(s).strip().upper() for s in v if str(s).strip()]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper

    signal_cutoff_et: str = Field(
        default="10:00",
        description=(
            "Latest ET time to generate new ORB signals (HH:MM format). "
            "Backtest window is 9:35-10:00; default aligns live with backtest."
        ),
    )
    adx_min_threshold: int = Field(
        default=25,
        description=(
            "Minimum daily ADX value to allow ORB signals. "
            "Backtest uses 25; lower values allow weaker trends."
        ),
    )
    orb_min_range_pct: float = Field(
        default=0.0015,
        description=(
            "Minimum ORB range as a fraction of price (0.0015 = 0.15%). "
            "Days with narrower ORBs are skipped (no signals)."
        ),
    )
    gap_threshold_pct: float = Field(
        default=0.3,
        description=(
            "Gap classification threshold (%). "
            "Gap > +threshold: LONG only; gap < -threshold: SHORT only; else both."
        ),
    )

    excluded_days: list[int] = Field(
        default=[0],
        description=(
            "Weekdays excluded from trading (0=Monday, 4=Friday). "
            "Shared by both live scanner and backtest engine."
        ),
    )

    execution_mode: str = Field(
        default="alerts_only",
        description="Execution mode: 'alerts_only', 'paper_trade', or 'live_trade'",
    )

    @field_validator("signal_cutoff_et")
    @classmethod
    def validate_signal_cutoff_et(cls, v: str) -> str:
        """Validate HH:MM format."""
        from datetime import time as _time

        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"signal_cutoff_et must be HH:MM, got {v!r}")
        _time(int(parts[0]), int(parts[1]))  # raises ValueError if invalid
        return v

    @field_validator("excluded_days", mode="before")
    @classmethod
    def parse_excluded_days(cls, v: str | list[int]) -> list[int]:
        """Accept comma-separated ints from .env (e.g. '0,4') or a list."""
        if isinstance(v, str):
            return [int(d.strip()) for d in v.split(",") if d.strip()]
        return [int(d) for d in v]

    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        valid = ("live", "paper", "backtest")
        if v not in valid:
            raise ValueError(f"trading_mode must be one of {valid}, got {v!r}")
        return v

    @field_validator("execution_mode")
    @classmethod
    def validate_execution_mode(cls, v: str) -> str:
        valid = ("alerts_only", "paper_trade", "live_trade")
        if v not in valid:
            raise ValueError(f"execution_mode must be one of {valid}, got {v!r}")
        return v


@lru_cache(maxsize=1)
def get_alpaca_settings() -> AlpacaSettings:
    return AlpacaSettings()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def get_telegram_settings() -> TelegramSettings:
    return TelegramSettings()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def get_risk_settings() -> RiskSettings:
    return RiskSettings()


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return AppSettings()
