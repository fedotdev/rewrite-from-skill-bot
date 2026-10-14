from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    bot_token: str = Field(..., validation_alias="BOT_TOKEN")
    allowed_chat_ids: list[int] = Field(default_factory=list, validation_alias="ALLOWED_CHAT_IDS")

    llm_api_url: str = Field(default="https://api.openai.com/v1/chat/completions", validation_alias="LLM_API_URL")
    llm_api_key: str = Field(..., validation_alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=30.0, validation_alias="LLM_TIMEOUT_SECONDS")
    llm_max_tokens: int = Field(default=2000, validation_alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.3, validation_alias="LLM_TEMPERATURE")
# response_format=json_object is NOT advertised by OpenRouter models and can 400; off by default
    llm_json_mode: bool = Field(default=False, validation_alias="LLM_JSON_MODE")

    rate_limit_per_user_per_minute: int = Field(default=20, validation_alias="RATE_LIMIT_PER_USER_PER_MINUTE")
    rate_limit_per_chat_per_minute: int = Field(default=100, validation_alias="RATE_LIMIT_PER_CHAT_PER_MINUTE")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
