from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "DocFlow"
    debug: bool = False
    google_api_key: str
    database_url: str
    chroma_persist_dir: str = "./chroma_db"
    mlflow_tracking_uri: str = "./mlflow_runs"
    port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
