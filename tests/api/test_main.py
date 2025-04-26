from fastapi.testclient import TestClient
from src.api.main import app  # Import the FastAPI app instance
from config.settings import settings # Import settings to use project name

client = TestClient(app)

def test_read_root():
    """Test the root endpoint '/'."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": f"Welcome to the {settings.PROJECT_NAME}!"}

def test_health_check():
    """Test the health check endpoint '/health'."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Add more tests for other main app functionalities if needed
