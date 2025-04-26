import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

# --- IMPORTANT: Set TESTING env var BEFORE importing modules that use settings ---
# This ensures that settings.DATABASE_URI points to the test DB
os.environ['TESTING'] = 'True'
# -----------------------------------------------------------------------------

# Now import application modules AFTER setting the env var
from src.common.db import Base, get_db # Import Base for table creation and get_db for overriding
from src.api.main import app # Import the FastAPI app instance
from config.settings import settings # Import settings to get the TEST_DATABASE_URI

# Use the TEST_DATABASE_URI from settings
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URI

# Create a new engine instance for testing
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # For SQLite, connect_args is needed to disable same-thread check
    connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)

# Create a sessionmaker instance for testing
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """
    Fixture to create and drop the test database schema once per session.
    'autouse=True' ensures it runs automatically for the session.
    """
    # Ensure all models inheriting from Base are imported before create_all
    # This might require importing specific model files if they aren't implicitly imported elsewhere
    import src.models.ingestion_log
    print("\nINFO: Creating test database tables...")
    Base.metadata.create_all(bind=engine)
    print("INFO: Test database tables created.")
    yield # Test session runs here
    print("\nINFO: Dropping test database tables...")
    Base.metadata.drop_all(bind=engine)
    print("INFO: Test database tables dropped.")


@pytest.fixture(scope="function")
def db_session() -> Session:
    """
    Provides a transactional database session for each test function.
    Rolls back changes after each test.
    """
    connection = engine.connect()
    # begin a non-ORM transaction
    transaction = connection.begin()
    # bind an individual Session to the connection
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    # rollback - everything that happened with the
    # Session above (including calls to commit())
    # is rolled back.
    transaction.rollback()
    # return connection to the Engine connection pool
    connection.close()


@pytest.fixture(scope="function")
def test_client(db_session: Session) -> TestClient:
    """
    Provides a TestClient instance configured with an overridden DB session dependency.
    """
    def override_get_db():
        """Override dependency to use the test session."""
        try:
            yield db_session
        finally:
            # The db_session fixture handles rollback/close
            pass

    # Override the get_db dependency in the FastAPI app
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    # Clean up the override after the test
    app.dependency_overrides.clear()
