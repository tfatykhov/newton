"""Settings via pydantic-settings with NOUS_ env prefix.

DB connection fields use validation_alias to read from the same unprefixed
env vars (DB_PASSWORD, DB_PORT, etc.) that docker-compose uses, so a single
.env file drives both the container and the Python app.
"""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOUS_", env_file=".env")

    # DB connection — unprefixed aliases match docker-compose env vars
    db_host: str = Field("localhost", validation_alias="DB_HOST")
    db_port: int = Field(5432, validation_alias="DB_PORT")
    db_user: str = Field("nous", validation_alias="DB_USER")
    db_password: str = Field("nous_dev_password", validation_alias="DB_PASSWORD")
    db_name: str = Field("nous", validation_alias="DB_NAME")

    db_pool_size: int = 10
    db_max_overflow: int = 5
    agent_id: str = "nous-default"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    log_level: str = "info"

    # Brain settings
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")
    auto_link_threshold: float = 0.85
    auto_link_max: int = 3
    quality_block_threshold: float = 0.5

    # Runtime
    host: str = "0.0.0.0"
    port: int = 8000
    anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")
    # Dual auth: auth_token (Bearer) takes precedence over api_key (x-api-key)
    anthropic_auth_token: str = Field("", validation_alias="ANTHROPIC_AUTH_TOKEN")

    # Agent identity
    agent_name: str = "Nous"
    agent_description: str = "A thinking agent that learns from experience"
    identity_prompt: str = ""

    # Event Bus
    event_bus_enabled: bool = True
    episode_summary_enabled: bool = True
    fact_extraction_enabled: bool = True
    sleep_enabled: bool = True
    background_model: str = Field(
        default="claude-sonnet-4-5-20250514",
        validation_alias="NOUS_BACKGROUND_MODEL",
    )
    session_idle_timeout: int = Field(
        default=1800,
        validation_alias="NOUS_SESSION_TIMEOUT",
    )
    sleep_timeout: int = Field(
        default=7200,
        validation_alias="NOUS_SLEEP_TIMEOUT",
    )
    sleep_check_interval: int = Field(
        default=60,
        validation_alias="NOUS_SLEEP_CHECK_INTERVAL",
    )

    # MCP
    mcp_enabled: bool = True

    # LLM
    model: str = "claude-sonnet-4-5-20250514"
    max_tokens: int = 4096

    # Extended thinking
    thinking_mode: Literal["off", "adaptive", "manual"] = "off"
    thinking_budget: int = 10000  # budget_tokens for manual mode (min 1024)
    effort: Literal["low", "medium", "high", "max"] = "high"

    # Direct API settings
    max_turns: int = 10  # Max tool use iterations per turn
    api_base_url: str = "https://api.anthropic.com"
    api_timeout_connect: int = 10  # seconds
    api_timeout_read: int = 120  # seconds
    workspace_dir: str = "/tmp/nous-workspace"

    # Web tools
    brave_search_api_key: str = Field("", validation_alias="BRAVE_SEARCH_API_KEY")
    web_search_daily_limit: int = 100  # Max web searches per day
    web_fetch_max_chars: int = 10000  # Default max chars for web_fetch

    @model_validator(mode="after")
    def _validate_thinking(self) -> "Settings":
        if self.thinking_mode == "manual":
            if self.thinking_budget < 1024:
                raise ValueError("thinking_budget must be >= 1024 (API minimum)")
            if self.thinking_budget >= self.max_tokens:
                raise ValueError(
                    f"thinking_budget ({self.thinking_budget}) must be < "
                    f"max_tokens ({self.max_tokens}). Increase max_tokens."
                )
        return self

    @property
    def db_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
