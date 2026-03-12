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

    @field_validator("risk_per_trade_pct", "max_daily_loss_pct")
    @classmethod
    def validate_pct(cls, v: Decimal) -> Decimal:
        if v <= 0 or v > 100:
            raise ValueError(f"percentage must be between 0 and 100, got {v}")
        return v

    @field_validator("max_trades_per_day")
    @classmethod
    def validate_max_trades(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_trades_per_day must be >= 1, got {v}")
        return v


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = Field(default="INFO", description="Log level")
    trading_mode: str = Field(
        default="paper", description="Trading mode: 'live', 'paper', or 'backtest'"
    )
    db_path: str = Field(default="data/spy_signals.db", description="Path to SQLite database")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper

    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        valid = ("live", "paper", "backtest")
        if v not in valid:
            raise ValueError(f"trading_mode must be one of {valid}, got {v!r}")
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
