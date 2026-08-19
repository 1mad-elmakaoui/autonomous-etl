"""Environment-driven configuration. No secret ever appears in source.

Loaded once per process. Temporal *activities* read this; Temporal *workflows*
must not (reading the environment inside workflow code is non-deterministic).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from etl_migrator.domain.errors import ConfigurationError


class LLMProvider(StrEnum):
    SCRIPTED = "scripted"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ETLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.SCRIPTED
    llm_model: str = "claude-sonnet-4-5"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tool_iterations: int = Field(default=6, ge=1, le=25)
    llm_fixture_dir: Path = Path("fixtures/llm")

    # --- Workspace ---------------------------------------------------------
    workspace_dir: Path = Path(".workspace")

    # --- Temporal ------------------------------------------------------------
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "etl-migration"
    temporal_execution_task_queue: str = Field(
        default="",
        description="Task queue for the activities that execute untrusted generated "
        "code. Empty means one queue for everything, which is right on a laptop. "
        "Set it and those activities are served by a separate worker, which is what "
        "lets Kubernetes give that worker no internet egress at all (phase 8).",
    )

    # --- Storage -------------------------------------------------------------
    postgres_url: str = "postgresql://etlm:etlm@localhost:5432/etlm"
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: SecretStr | None = None
    minio_secret_key: SecretStr | None = None
    minio_bucket: str = "etl-migrations"

    # --- GitHub --------------------------------------------------------------
    github_token: SecretStr | None = None
    github_repository: str | None = None

    # --- Kubernetes ----------------------------------------------------------
    kubernetes_namespace: str = "etl-migration"

    # --- Observability -----------------------------------------------------
    log_level: str = "INFO"
    log_format: str = Field(default="console", description="'console' or 'json'")
    metrics_port: int = Field(default=9464, ge=1, le=65535)

    @field_validator("log_format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in {"console", "json"}:
            raise ValueError("log_format must be 'console' or 'json'")
        return v

    def require_llm_credentials(self) -> str:
        """Return the API key, or explain precisely what is missing."""
        if self.llm_provider is LLMProvider.SCRIPTED:
            raise ConfigurationError("scripted provider has no credentials")
        if self.llm_api_key is None or not self.llm_api_key.get_secret_value():
            raise ConfigurationError(
                f"ETLM_LLM_API_KEY is required when ETLM_LLM_PROVIDER={self.llm_provider.value}. "
                "Set it in .env (see .env.example) or export it."
            )
        return self.llm_api_key.get_secret_value()


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton.

    `refresh=True` is used by tests that monkeypatch the environment.
    """
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
