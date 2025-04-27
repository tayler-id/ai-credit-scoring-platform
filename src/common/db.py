from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from fastapi import HTTPException, status
from config.settings import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URI

# Always define Base so ORM models can be imported even if DB is down
Base = declarative_base()

try:
    # Adjust pool settings as needed for production
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True # Checks connections for liveness before handing them out
        # pool_size=10, # Example: Set pool size
        # max_overflow=20 # Example: Set max overflow
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info(f"Database engine created successfully for URI: {settings.DATABASE_URI}")

    # Optional: Test connection on startup
    with engine.connect() as connection:
        logger.info("Database connection test successful.")

except Exception as e:
    logger.error(f"Failed to create database engine or connect: {e}", exc_info=True)
    # Handle initialization error appropriately - maybe prevent app startup
    engine = None
    SessionLocal = None
    # Do NOT set Base = None here!

# Dependency function to get a DB session for FastAPI endpoints
def get_db():
    """
    FastAPI dependency that provides a SQLAlchemy database session.
    Ensures the session is closed after the request.
    """
    if SessionLocal is None:
        logger.error("Database session is not initialized. Cannot get DB session.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not available."
        )

    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Error during database session: {e}", exc_info=True)
        db.rollback() # Rollback in case of error during the request handling
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during database operation."
        )
    finally:
        db.close()

logger.info(f"Database module loaded. Engine and SessionLocal configured.")

# Note: The Base object should be imported by your ORM models (e.g., in src/models/orm.py)
# Example: from src.common.db import Base
