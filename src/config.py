"""
Configuration management for Trading Intelligence Hub
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class APIConfig(BaseSettings):
    """API Configuration"""
    api_key: str = Field(validation_alias="API_KEY")
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class PerplexityConfig(BaseSettings):
    """Perplexity Finance API Configuration"""
    api_key: str = Field(validation_alias="PPLX_FINANCE_API_KEY")
    base_url: str = Field(default="https://api.perplexity.ai", validation_alias="PPLX_FINANCE_BASE")
    timeout: int = Field(default=30, validation_alias="PPLX_TIMEOUT")
    model: str = "sonar-pro"
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class RedisConfig(BaseSettings):
    """Redis Configuration"""
    host: str = Field(default="localhost", validation_alias="REDIS_HOST")
    port: int = Field(default=6379, validation_alias="REDIS_PORT")
    password: Optional[str] = None
    db: int = Field(default=0, validation_alias="REDIS_DB")
    channel_trades: str = "trades.exec"
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class MT5Config(BaseSettings):
    """MetaTrader5 Configuration"""
    login: str = Field(validation_alias="MT5_LOGIN")
    password: str = Field(validation_alias="MT5_PASSWORD")
    server: str = Field(validation_alias="MT5_SERVER")
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class TradingConfig(BaseSettings):
    """Trading Configuration"""
    risk_per_trade_pct: float = 1.0
    daily_max_drawdown_pct: float = 5.0
    max_positions: int = 5
    min_equity: float = 1000.0
    initial_equity: float = 10000.0
    max_leverage: float = 1.0
    reward_risk_ratio: float = 2.0
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class ChartConfig(BaseSettings):
    """Chart Analysis Configuration"""
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False, extra="ignore")


class StrategyConfig(BaseSettings):
    """Strategy Configuration"""
    sentiment_weight: float = 0.3
    chart_weight: float = 0.4
    swarm_weight: float = 0.3
    min_consensus_score: float = 0.6
    
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False, extra="ignore")


class MarketDataConfig(BaseSettings):
    """Market Data Configuration"""
    crypto_exchange: str = "binance"
    stock_provider: str = "yfinance"
    update_interval: int = 60
    
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False, extra="ignore")


class LoggingConfig(BaseSettings):
    """Logging Configuration"""
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: str = "json"
    log_file: str = Field(default="logs/trading_hub.log", validation_alias="LOG_FILE")
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class SecurityConfig(BaseSettings):
    """Security Configuration"""
    jwt_secret_key: str = Field(validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=1440, validation_alias="JWT_EXPIRE_MINUTES")
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class AppConfig(BaseSettings):
    """Application Configuration"""
    debug: bool = False
    testing: bool = False
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class Config:
    """Main configuration class that aggregates all configs"""
    
    def __init__(self):
        self.api = APIConfig()
        self.perplexity = PerplexityConfig()
        self.redis = RedisConfig()
        self.mt5 = MT5Config()
        self.trading = TradingConfig()
        self.chart = ChartConfig()
        self.strategy = StrategyConfig()
        self.market_data = MarketDataConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.app = AppConfig()
    
    def validate(self) -> bool:
        """Validate all configurations"""
        try:
            # Validate required fields are present
            required_configs = [
                self.api.api_key,
                self.perplexity.api_key,
                self.mt5.login,
                self.mt5.password,
                self.mt5.server,
                self.security.jwt_secret_key
            ]
            
            for config in required_configs:
                if not config or config == "your_key_here" or "your_" in config:
                    return False
            
            # Validate numeric ranges
            if not (0 < self.trading.risk_per_trade_pct <= 10):
                return False
            
            if not (0 < self.trading.daily_max_drawdown_pct <= 50):
                return False
            
            if not (0.1 <= self.strategy.min_consensus_score <= 1.0):
                return False
            
            return True
            
        except Exception:
            return False


_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance lazily"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def is_production() -> bool:
    """Check if running in production mode"""
    cfg = get_config()
    return not cfg.app.debug and not cfg.app.testing


def is_testing() -> bool:
    """Check if running in testing mode"""
    cfg = get_config()
    return cfg.app.testing