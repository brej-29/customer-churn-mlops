from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env so the _read_genai_env validator can see
# bare vars like GEMINI_API_KEY (pydantic-settings only maps CHURN_* names).
load_dotenv(override=False)  # does not clobber existing env vars


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    random_seed: int = 42

    # Data paths
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    telco_csv_path: Path = Path("data/raw/telco_churn.csv")

    # MLflow — SQLite backend; MLflow 3.x dropped the file-store backend.
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"

    # GenAI / LLM settings (Tier 2)
    llm_provider: Literal["gemini", "groq"] = "gemini"
    llm_model: str | None = None
    # Read from GEMINI_API_KEY / CHURN_GEMINI_API_KEY (see _read_genai_env below).
    gemini_api_key: str | None = Field(default=None, repr=False)
    groq_api_key: str | None = Field(default=None, repr=False)
    # None = auto-detect from key presence; set CHURN_EXPLANATION_ENABLED to override.
    explanation_enabled: bool | None = None
    rag_corpus_path: Path = Path("data/playbooks")

    @field_validator(
        "data_raw_dir", "data_processed_dir", "telco_csv_path", "rag_corpus_path",
        mode="before",
    )
    @classmethod
    def coerce_path(cls, v: object) -> Path:
        return Path(str(v))

    @model_validator(mode="before")
    @classmethod
    def _read_genai_env(cls, data: Any) -> Any:
        """Accept bare env vars (GEMINI_API_KEY etc.) alongside the CHURN_* prefix."""
        import os
        if not isinstance(data, dict):
            return data
        for field, env_var in [
            ("gemini_api_key", "GEMINI_API_KEY"),
            ("groq_api_key", "GROQ_API_KEY"),
            ("llm_provider", "LLM_PROVIDER"),
            ("llm_model", "LLM_MODEL"),
            # Bare MLFLOW_TRACKING_URI is the standard MLflow env var; CHURN_ prefix also works.
            # Auth: set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD — MLflow reads
            # those natively, so they don't need to appear in Settings.
            ("mlflow_tracking_uri", "MLFLOW_TRACKING_URI"),
        ]:
            if not data.get(field):
                val = os.environ.get(env_var)
                if val is not None:
                    data[field] = val
        return data

    @model_validator(mode="after")
    def _resolve_explanation_enabled(self) -> "Settings":
        """Auto-enable explanations when at least one LLM API key is configured."""
        if self.explanation_enabled is None:
            self.explanation_enabled = bool(self.gemini_api_key or self.groq_api_key)
        return self


settings = Settings()
