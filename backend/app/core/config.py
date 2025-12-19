from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str = "dev-secret"
    JWT_SECRET: str = "dev-jwt-secret"
    ALGORITHM: str = "HS256"

    ALLOWED_ORIGINS: List[str] = []
    ALLOWED_HOSTS: List[str] = []
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    AUTH_PREFIX: str = 'Bearer '

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
