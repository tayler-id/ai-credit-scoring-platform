import pytest
from fastapi.testclient import TestClient
from src.api.main import app # Import the FastAPI app instance
from config.settings import settings # Import settings to construct the URL prefix

# Note: We use the test_client fixture automatically from conftest.py
# which handles DB setup and dependency overrides.

API_PREFIX = settings.API_V1_STR

def test_request_score_success(test_client: TestClient):
    """Test successfully requesting a score calculation."""
    applicant_id = "score_test_01"
    request_payload = {"applicant_id": applicant_id}

    response = test_client.post(f"{API_PREFIX}/score", json=request_payload)

    assert response.status_code == 202 # Accepted
    response_json = response.json()
    assert response_json["applicant_id"] == applicant_id
    assert response_json["status"] == "processing" # Check initial status
    assert "message" in response_json
    # We don't expect score or risk level yet in this placeholder implementation

def test_request_score_missing_applicant_id(test_client: TestClient):
    """Test score request with missing applicant_id."""
    request_payload = {} # Missing applicant_id
    response = test_client.post(f"{API_PREFIX}/score", json=request_payload)
    assert response.status_code == 422 # Unprocessable Entity

# Add more tests later when the scoring logic is implemented,
# potentially involving mocking the background task or checking results
# via a separate status endpoint.
