import uuid
from fastapi.testclient import TestClient
from src.api.main import app # Import the FastAPI app instance
from config.settings import settings # Import settings to construct the URL prefix

client = TestClient(app)

API_PREFIX = settings.API_V1_STR

def test_ingest_data_success():
    """Test successful data ingestion."""
    test_payload = {
        "applicant_id": "test_applicant_123",
        "data_source": "test_source",
        "payload": {"key1": "value1", "key2": 123}
    }
    response = client.post(f"{API_PREFIX}/ingest", json=test_payload)

    assert response.status_code == 202 # 202 Accepted
    response_json = response.json()
    assert "ingestion_id" in response_json
    assert "status" in response_json
    assert response_json["status"] == "received"
    # Check if ingestion_id is a valid UUID
    try:
        uuid.UUID(response_json["ingestion_id"])
        valid_uuid = True
    except ValueError:
        valid_uuid = False
    assert valid_uuid

def test_ingest_data_missing_field():
    """Test ingestion request with a missing required field (applicant_id)."""
    test_payload = {
        # "applicant_id": "test_applicant_123", # Missing
        "data_source": "test_source",
        "payload": {"key1": "value1"}
    }
    response = client.post(f"{API_PREFIX}/ingest", json=test_payload)
    # FastAPI/Pydantic handles validation automatically
    assert response.status_code == 422 # Unprocessable Entity

def test_ingest_data_invalid_payload_type():
    """Test ingestion request with invalid payload type (should be dict)."""
    test_payload = {
        "applicant_id": "test_applicant_456",
        "data_source": "test_source",
        "payload": "this is not a dictionary" # Invalid type
    }
    response = client.post(f"{API_PREFIX}/ingest", json=test_payload)
    assert response.status_code == 422 # Unprocessable Entity

# Add more tests as needed, e.g., for different data sources or edge cases
