import pytest
import uuid
from sqlalchemy.orm import Session
from src.crud import crud_ingestion_log
from src.common.models import AlternativeDataPayload
from src.models.ingestion_log import IngestionLog # Import the ORM model

# Note: We don't need to import fixtures like db_session, pytest finds them automatically from conftest.py

def test_create_ingestion_log(db_session: Session):
    """Test creating an ingestion log entry."""
    test_uuid = uuid.uuid4()
    payload_in = AlternativeDataPayload(
        applicant_id="crud_test_01",
        data_source="test_source_crud",
        payload={"data": "some test data"}
    )
    db_obj = crud_ingestion_log.create_ingestion_log(db=db_session, ingestion_id=test_uuid, payload_in=payload_in)

    assert db_obj is not None
    assert db_obj.id == test_uuid
    assert db_obj.applicant_id == "crud_test_01"
    assert db_obj.data_source == "test_source_crud"
    assert db_obj.payload == {"data": "some test data"}
    assert db_obj.status == "received"
    assert db_obj.received_at is not None

    # Verify it's actually in the DB (within the same transaction)
    retrieved_obj = db_session.query(IngestionLog).filter(IngestionLog.id == test_uuid).first()
    assert retrieved_obj is not None
    assert retrieved_obj.id == test_uuid

def test_get_ingestion_log(db_session: Session):
    """Test retrieving an ingestion log entry by ID."""
    # First, create an entry to retrieve
    test_uuid = uuid.uuid4()
    payload_in = AlternativeDataPayload(applicant_id="crud_test_02", data_source="get_test", payload={"a": 1})
    crud_ingestion_log.create_ingestion_log(db=db_session, ingestion_id=test_uuid, payload_in=payload_in)

    # Now retrieve it
    retrieved_obj = crud_ingestion_log.get_ingestion_log(db=db_session, ingestion_id=test_uuid)
    assert retrieved_obj is not None
    assert retrieved_obj.id == test_uuid
    assert retrieved_obj.applicant_id == "crud_test_02"

def test_get_ingestion_log_not_found(db_session: Session):
    """Test retrieving a non-existent ingestion log entry."""
    non_existent_uuid = uuid.uuid4()
    retrieved_obj = crud_ingestion_log.get_ingestion_log(db=db_session, ingestion_id=non_existent_uuid)
    assert retrieved_obj is None

def test_get_ingestion_logs_by_applicant(db_session: Session):
    """Test retrieving logs for a specific applicant."""
    applicant = "crud_test_03"
    # Create multiple entries for the same applicant
    payload1 = AlternativeDataPayload(applicant_id=applicant, data_source="src1", payload={"p": 1})
    payload2 = AlternativeDataPayload(applicant_id=applicant, data_source="src2", payload={"p": 2})
    payload_other = AlternativeDataPayload(applicant_id="other_applicant", data_source="src3", payload={"p": 3})

    crud_ingestion_log.create_ingestion_log(db=db_session, ingestion_id=uuid.uuid4(), payload_in=payload1)
    crud_ingestion_log.create_ingestion_log(db=db_session, ingestion_id=uuid.uuid4(), payload_in=payload2)
    crud_ingestion_log.create_ingestion_log(db=db_session, ingestion_id=uuid.uuid4(), payload_in=payload_other)

    # Retrieve logs for 'crud_test_03'
    logs = crud_ingestion_log.get_ingestion_logs_by_applicant(db=db_session, applicant_id=applicant)
    assert len(logs) == 2
    assert all(log.applicant_id == applicant for log in logs)
    # Check default ordering (descending by received_at) - harder to assert precisely without mocking time
    assert logs[0].data_source in ["src1", "src2"]
    assert logs[1].data_source in ["src1", "src2"]
    assert logs[0].data_source != logs[1].data_source

    # Test pagination (limit)
    logs_limit_1 = crud_ingestion_log.get_ingestion_logs_by_applicant(db=db_session, applicant_id=applicant, limit=1)
    assert len(logs_limit_1) == 1

    # Test pagination (skip) - assuming limit=100 default
    logs_skip_1 = crud_ingestion_log.get_ingestion_logs_by_applicant(db=db_session, applicant_id=applicant, skip=1)
    assert len(logs_skip_1) == 1

def test_update_ingestion_log_status(db_session: Session):
    """Test updating the status of an ingestion log."""
    test_uuid = uuid.uuid4()
    payload_in = AlternativeDataPayload(applicant_id="crud_test_04", data_source="update_test", payload={"status": "initial"})
    created_obj = crud_ingestion_log.create_ingestion_log(db=db_session, ingestion_id=test_uuid, payload_in=payload_in)
    assert created_obj.status == "received"

    # Update the status
    updated_obj = crud_ingestion_log.update_ingestion_log_status(db=db_session, ingestion_id=test_uuid, status="processed")
    assert updated_obj is not None
    assert updated_obj.id == test_uuid
    assert updated_obj.status == "processed"

    # Verify the change in the database
    retrieved_obj = crud_ingestion_log.get_ingestion_log(db=db_session, ingestion_id=test_uuid)
    assert retrieved_obj is not None
    assert retrieved_obj.status == "processed"

def test_update_ingestion_log_status_not_found(db_session: Session):
    """Test updating the status of a non-existent log."""
    non_existent_uuid = uuid.uuid4()
    updated_obj = crud_ingestion_log.update_ingestion_log_status(db=db_session, ingestion_id=non_existent_uuid, status="error")
    assert updated_obj is None
