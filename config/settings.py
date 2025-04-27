import os
import os
from typing import Optional, Union
from pydantic_settings import BaseSettings # Updated import
from pydantic import PostgresDsn, validator # Keep these if still used
from dotenv import load_dotenv # Keep this import if needed elsewhere, but remove the call below

# Load .env file variables - pydantic-settings handles this via Config

class Settings(BaseSettings):
    """Application settings."""

    PROJECT_NAME: str = "AI Credit Scoring Platform"
    API_V1_STR: str = "/api/v1"

    # --- Database Settings ---
    # Use environment variable to switch between prod/dev and test databases
    TESTING: bool = os.getenv("TESTING", "False").lower() in ("true", "1", "t")

    # Production/Development Database (PostgreSQL)
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432")) # Changed type hint to int and added explicit cast
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "credit_scoring_db")

    # Test Database (Defaults to SQLite in-memory if TEST_DATABASE_URI not set)
    TEST_DATABASE_URI: Optional[str] = os.getenv("TEST_DATABASE_URI", "sqlite+pysqlite:///:memory:") # Use pysqlite driver

    DATABASE_URI: Optional[str] = None

    @validator("DATABASE_URI", pre=True, always=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            # If DATABASE_URI is explicitly set (e.g., in .env), use it
            return v
        if values.get("TESTING"):
            # Use test database URI if TESTING is True
            print("INFO: Using TEST database URI:", values.get("TEST_DATABASE_URI")) # Added print for visibility
            return values.get("TEST_DATABASE_URI")
        else:
            # Assemble PostgreSQL URI for prod/dev
            pg_uri = PostgresDsn.build(
                scheme="postgresql+psycopg2", # Specify psycopg2 driver
                username=values.get("POSTGRES_USER"), # Corrected from 'user' to 'username'
                password=values.get("POSTGRES_PASSWORD"),
                host=values.get("POSTGRES_SERVER"),
                port=values.get("POSTGRES_PORT"),
                path=f"{values.get('POSTGRES_DB') or ''}",
            )
            print("INFO: Using PostgreSQL database URI:", pg_uri) # Added print for visibility
            return str(pg_uri)

    # --- AWS Settings (Example) ---
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "ai-credit-scoring-data")

    # Add other settings as needed (e.g., MLflow tracking URI, external API keys)

    class Config:
        case_sensitive = True
        # If you have a .env file, settings will be loaded from it
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# You can access settings like this:
# from config.settings import settings
# print(settings.PROJECT_NAME)
