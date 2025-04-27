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

    assert response.status_code == 200  # Now synchronous, should return 200 OK
    response_json = response.json()
    assert response_json["applicant_id"] == applicant_id
    assert response_json["status"] == "completed"
    assert "message" in response_json
    assert "score" in response_json
    assert "risk_level" in response_json
    assert response_json["score"] is not None
    assert response_json["risk_level"] in {"low", "medium", "high"}

def test_request_score_missing_applicant_id(test_client: TestClient):
    """Test score request with missing applicant_id."""
    request_payload = {} # Missing applicant_id
    response = test_client.post(f"{API_PREFIX}/score", json=request_payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_request_score_mobile_money_only(test_client: TestClient):
    """
    Test score calculation for an applicant with only mobile money data (no utility payments).
    The mock_fetch_mobile_money_transactions will return mock data for any applicant_id.
    """
    applicant_id = "mm_only_test_01"
    request_payload = {"applicant_id": applicant_id}

    response = test_client.post(f"{API_PREFIX}/score", json=request_payload)

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["applicant_id"] == applicant_id
    assert response_json["status"] == "completed"
    assert "score" in response_json
    assert "risk_level" in response_json
    # The score should be > 0.2 (the minimum for no data), since mobile money data is present
    assert response_json["score"] > 0.2

def test_request_score_ecommerce_only(test_client: TestClient, monkeypatch):
    """
    Test score calculation for an applicant with only e-commerce data (no utility or mobile money).
    Patch mobile money fetch to return empty DataFrame, and ensure no utility payments in DB.
    """
    import pandas as pd
    from src.data_ingestion import ecommerce, mobile_money

    applicant_id = "ec_only_test_01"
    request_payload = {"applicant_id": applicant_id}

    # Patch mobile money fetch to return empty DataFrame
    monkeypatch.setattr(mobile_money, "mock_fetch_mobile_money_transactions", lambda x: pd.DataFrame())
    # Patch e-commerce fetch to return mock data (already does by default)

    response = test_client.post(f"{API_PREFIX}/score", json=request_payload)

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["applicant_id"] == applicant_id
    assert response_json["status"] == "completed"
    assert "score" in response_json
    assert "risk_level" in response_json
    # The score should be > 0.2 (the minimum for no data), since e-commerce data is present
    assert response_json["score"] > 0.2

def test_request_score_all_sources(test_client: TestClient):
    """
    Test score calculation for an applicant with utility, mobile money, and e-commerce data.
    All mock fetchers return data by default for a new applicant_id.
    """
    applicant_id = "all_sources_test_01"
    request_payload = {"applicant_id": applicant_id}

    response = test_client.post(f"{API_PREFIX}/score", json=request_payload)

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["applicant_id"] == applicant_id
    assert response_json["status"] == "completed"
    assert "score" in response_json
    assert "risk_level" in response_json
    # The score should be > 0.2 (the minimum for no data), since all data sources are present
    assert response_json["score"] > 0.2

# Add more tests later when the scoring logic is implemented,
# potentially involving mocking the background task or checking results
# via a separate status endpoint.
