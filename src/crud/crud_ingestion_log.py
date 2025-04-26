from sqlalchemy.orm import Session
from src.models.ingestion_log import IngestionLog
from src.common.models import AlternativeDataPayload # Pydantic model for type hinting if needed
import uuid
import logging

logger = logging.getLogger(__name__)

def create_ingestion_log(db: Session, *, ingestion_id: uuid.UUID, payload_in: AlternativeDataPayload) -> IngestionLog:
    """
    Creates a new ingestion log record in the database.

    Args:
        db: SQLAlchemy database session.
        ingestion_id: The pre-generated UUID for this ingestion event.
        payload_in: The Pydantic model containing the input data.

    Returns:
        The created IngestionLog ORM object.
    """
    db_obj = IngestionLog(
        id=ingestion_id,
        applicant_id=payload_in.applicant_id,
        data_source=payload_in.data_source,
        payload=payload_in.payload,
        status="received"
    )
    try:
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        logger.info(f"Created ingestion log with ID: {db_obj.id}")
        return db_obj
    except Exception as e:
        logger.error(f"Error creating ingestion log {ingestion_id}: {e}", exc_info=True)
        db.rollback()
        raise # Re-raise the exception to be handled by the caller (e.g., the API endpoint)

def get_ingestion_log(db: Session, ingestion_id: uuid.UUID) -> IngestionLog | None:
    """
    Retrieves an ingestion log record by its ID.

    Args:
        db: SQLAlchemy database session.
        ingestion_id: The UUID of the ingestion log to retrieve.

    Returns:
        The IngestionLog ORM object if found, otherwise None.
    """
    return db.query(IngestionLog).filter(IngestionLog.id == ingestion_id).first()

def get_ingestion_logs_by_applicant(db: Session, applicant_id: str, skip: int = 0, limit: int = 100) -> list[IngestionLog]:
    """
    Retrieves ingestion logs for a specific applicant with pagination.

    Args:
        db: SQLAlchemy database session.
        applicant_id: The applicant's identifier.
        skip: Number of records to skip (for pagination).
        limit: Maximum number of records to return (for pagination).

    Returns:
        A list of IngestionLog ORM objects.
    """
    return db.query(IngestionLog)\
             .filter(IngestionLog.applicant_id == applicant_id)\
             .order_by(IngestionLog.received_at.desc())\
             .offset(skip)\
             .limit(limit)\
             .all()

def update_ingestion_log_status(db: Session, ingestion_id: uuid.UUID, status: str) -> IngestionLog | None:
    """
    Updates the status of an existing ingestion log record.

    Args:
        db: SQLAlchemy database session.
        ingestion_id: The UUID of the ingestion log to update.
        status: The new status string (e.g., 'processing', 'processed', 'error').

    Returns:
        The updated IngestionLog ORM object if found and updated, otherwise None.
    """
    db_obj = get_ingestion_log(db, ingestion_id)
    if db_obj:
        db_obj.status = status
        try:
            db.commit()
            db.refresh(db_obj)
            logger.info(f"Updated status for ingestion log {ingestion_id} to '{status}'")
            return db_obj
        except Exception as e:
            logger.error(f"Error updating status for ingestion log {ingestion_id}: {e}", exc_info=True)
            db.rollback()
            raise
    return None

# Add other CRUD functions as needed (e.g., delete, complex queries)
