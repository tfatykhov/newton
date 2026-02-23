"""Settings via pydantic-settings with NOUS_ env prefix.

DB connection fields use validation_alias to read from the same unprefixed
env vars (DB_PASSWORD, DB_PORT, etc.) that docker-compose uses, so a single
.env file drives both the container and the Python app.
"""

from pydantic import Field
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
    # Dual auth: auth_token takes precedence over api_key (SDK uses auth_token)
    anthropic_auth_token: str = Field("", validation_alias="ANTHROPIC_AUTH_TOKEN")

    # Agent identity
    agent_name: str = "Nous"
    agent_description: str = "A thinking agent that learns from experience"
    identity_prompt: str = ""

    # MCP
    mcp_enabled: bool = True

    # LLM
    model: str = "claude-sonnet-4-5-20250514"
    max_tokens: int = 4096

    # SDK settings (claude-agent-sdk)
    sdk_max_turns: int = 10  # Max tool use iterations per query
    sdk_permission_mode: str = "default"  # default, acceptEdits, plan, bypassPermissions
    sdk_workspace: str = "/tmp/nous-workspace"
    sdk_allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "record_decision",
            "learn_fact",
            "recall_deep",
            "create_censor",
        ]
    )

    @property
    def db_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
