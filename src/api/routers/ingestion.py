from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from src.common.models import AlternativeDataPayload, IngestionResponse
from src.common.db import get_db
from src.crud import crud_ingestion_log
from src.data_processing.processor import process_data_for_applicant # Import processor
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest Alternative Data",
    description="Receives a payload of alternative data for a specific applicant and data source.",
    tags=["Ingestion"]
)
async def ingest_data(
    payload: AlternativeDataPayload,
    background_tasks: BackgroundTasks, # Add BackgroundTasks dependency
    db: Session = Depends(get_db)
):
    """
    Endpoint to ingest alternative data.

    - **applicant_id**: Unique identifier for the applicant.
    - **data_source**: Identifier for the source of the data.
    - **payload**: The actual alternative data payload (JSON object).

    This endpoint currently acknowledges receipt and logs the data.
    Future implementations might trigger background processing after saving.
    """
    ingestion_id = uuid.uuid4() # Generate ID for this specific ingestion event
    logger.info(f"Received data ingestion request: id={ingestion_id}, applicant_id={payload.applicant_id}, source={payload.data_source}")

    try:
        # Use the CRUD function to create the log entry
        db_ingestion_log = crud_ingestion_log.create_ingestion_log(
            db=db, ingestion_id=ingestion_id, payload_in=payload
        )
        logger.info(f"Successfully saved ingestion log via CRUD: id={ingestion_id}")

        # --- Trigger background processing ---
        # Add the data processing task to run in the background
        # Note: For production, consider more robust task queues like Celery.
        # Pass necessary arguments (like applicant_id). DB session cannot be passed directly
        # to background tasks easily; the task itself should create its own session if needed.
        # For this placeholder, we'll assume process_data_for_applicant handles its session.
        # A better approach might be to pass IDs and let the task fetch data.
        background_tasks.add_task(process_data_for_applicant, db, payload.applicant_id)
        logger.info(f"Queued background data processing for applicant: {payload.applicant_id}")
        status_msg = "Data received and queued for processing."
        # -----------------------------------

    except Exception as e:
        # The CRUD function handles rollback and logging, just re-raise or handle API response
        logger.error(f"Failed to process ingestion request {ingestion_id} due to database error: {e}", exc_info=False) # Log less verbosely here as CRUD logs details
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save ingestion data due to a database error."
        )

    return IngestionResponse(
        ingestion_id=ingestion_id,
        status=db_ingestion_log.status, # Return the status from the saved record
        message=status_msg
    )

# You might add other ingestion-related endpoints here, e.g., checking ingestion status
