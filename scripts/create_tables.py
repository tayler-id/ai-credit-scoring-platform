import sys
import os
import logging

# Add the project root to the Python path to allow imports from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import necessary components AFTER adjusting sys.path
    from src.common.db import engine, Base
    # Import all models that need to be created
    from src.models.ingestion_log import IngestionLog
    from src.models.applicant_features import ApplicantFeatures
    # Add other model imports here if needed in the future
    # from src.models.another_model import AnotherModel

    logger.info("Attempting to create database tables...")

    if engine is None:
        logger.error("Database engine is not initialized. Cannot create tables. Check .env configuration and db connection.")
        sys.exit(1)

    # Create all tables defined that inherit from Base
    Base.metadata.create_all(bind=engine)

    logger.info("Database tables created successfully (if they didn't exist).")

except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}. Ensure the script is run from the project root or the path is correctly configured.", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred during table creation: {e}", exc_info=True)
    sys.exit(1)
