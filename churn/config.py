from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @field_validator("data_raw_dir", "data_processed_dir", "telco_csv_path", mode="before")
    @classmethod
    def coerce_path(cls, v: object) -> Path:
        return Path(str(v))


settings = Settings()
