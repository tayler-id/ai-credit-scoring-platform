import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from src.data_ingestion.utility_payments import ingest_utility_payments

# Sample valid utility payment record data
valid_record_data_1 = {
    "provider": "City Power",
    "account_number": "ACC123",
    "payment_date": "2024-03-15",
    "due_date": "2024-03-20",
    "amount_paid": 55.75,
    "currency": "USD",
    "payment_status": "paid_on_time",
    "bill_period_start": "2024-02-15",
    "bill_period_end": "2024-03-14",
}

valid_record_data_2 = {
    "provider": "Water Board",
    "payment_date": "2024-03-10",
    "amount_paid": 25.00,
    "currency": "USD",
    "payment_status": "paid_late",
}

# Sample invalid utility payment record data (missing required field 'provider')
invalid_record_data = {
    "account_number": "ACC456",
    "payment_date": "2024-03-18",
    "amount_paid": 30.00,
    "currency": "USD",
    "payment_status": "paid_on_time",
}

@pytest.fixture
def mock_logger():
    """Fixture to mock the logger used in the ingestion function."""
    with patch('src.data_ingestion.utility_payments.logger', new_callable=MagicMock) as mock_log:
        yield mock_log

def test_ingest_utility_payments_success(mock_logger):
    """Test successful ingestion of valid utility payment records."""
    applicant_id = "applicant-001"
    payload = {
        "utility_payments": [valid_record_data_1, valid_record_data_2]
    }

    ingest_utility_payments(applicant_id, payload)

    # Assert logger calls
    mock_logger.info.assert_any_call(f"Starting utility payment ingestion for applicant: {applicant_id}")
    mock_logger.info.assert_any_call(f"Successfully validated 2 utility payment records for applicant {applicant_id}.")
    mock_logger.info.assert_any_call(f"Completed utility payment ingestion process for applicant: {applicant_id}")
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()

def test_ingest_utility_payments_validation_error(mock_logger):
    """Test ingestion with one valid and one invalid record."""
    applicant_id = "applicant-002"
    payload = {
        "utility_payments": [valid_record_data_1, invalid_record_data]
    }

    ingest_utility_payments(applicant_id, payload)

    # Assert logger calls
    mock_logger.info.assert_any_call(f"Starting utility payment ingestion for applicant: {applicant_id}")
    # Check for the validation warning log
    mock_logger.warning.assert_any_call(pytest.approx("Validation failed for record 2 for applicant applicant-002")) # Use approx match for error details
    mock_logger.warning.assert_any_call(f"Found 1 validation errors during utility payment ingestion for applicant {applicant_id}.")
    mock_logger.info.assert_any_call(f"Successfully validated 1 utility payment records for applicant {applicant_id}.") # Only 1 valid
    mock_logger.info.assert_any_call(f"Completed utility payment ingestion process for applicant {applicant_id}")
    mock_logger.error.assert_not_called()


def test_ingest_utility_payments_invalid_payload_structure(mock_logger):
    """Test ingestion with an invalid payload structure (missing 'utility_payments' key)."""
    applicant_id = "applicant-003"
    payload = {
        "some_other_key": []
    }

    ingest_utility_payments(applicant_id, payload)

    # Assert logger calls
    mock_logger.info.assert_any_call(f"Starting utility payment ingestion for applicant: {applicant_id}")
    mock_logger.error.assert_called_once_with(f"Invalid payload structure for applicant {applicant_id}. Missing 'utility_payments' list.")
    # Ensure completion logs are not called in this error case
    assert mock_logger.info.call_count == 1 # Only the starting log

def test_ingest_utility_payments_invalid_payload_type(mock_logger):
    """Test ingestion with an invalid payload structure ('utility_payments' is not a list)."""
    applicant_id = "applicant-004"
    payload = {
        "utility_payments": {"not": "a list"}
    }

    ingest_utility_payments(applicant_id, payload)

    # Assert logger calls
    mock_logger.info.assert_any_call(f"Starting utility payment ingestion for applicant: {applicant_id}")
    mock_logger.error.assert_called_once_with(f"Invalid payload structure for applicant {applicant_id}. Missing 'utility_payments' list.")
     # Ensure completion logs are not called in this error case
    assert mock_logger.info.call_count == 1 # Only the starting log

def test_ingest_utility_payments_empty_list(mock_logger):
    """Test ingestion with an empty list of payments."""
    applicant_id = "applicant-005"
    payload = {
        "utility_payments": []
    }

    ingest_utility_payments(applicant_id, payload)

    # Assert logger calls
    mock_logger.info.assert_any_call(f"Starting utility payment ingestion for applicant: {applicant_id}")
    mock_logger.info.assert_any_call(f"Successfully validated 0 utility payment records for applicant {applicant_id}.")
    mock_logger.info.assert_any_call(f"Completed utility payment ingestion process for applicant {applicant_id}")
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()
