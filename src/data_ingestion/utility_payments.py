import logging
from typing import List, Dict, Any
from pydantic import ValidationError

# Import DB session management and CRUD function
from src.common.db import SessionLocal
from src.crud.crud_utility_payment import create_utility_payments_bulk
from src.common.models import UtilityPaymentRecord

logger = logging.getLogger(__name__)

def ingest_utility_payments(applicant_id: str, payload: Dict[str, Any]):
    """
    Handles the ingestion and initial validation of utility payment data.

    Args:
        applicant_id: The ID of the applicant.
        payload: The raw data payload, expected to contain a list of utility records.
                 Example structure: {"utility_payments": [{...}, {...}]}
    """
    logger.info(f"Starting utility payment ingestion for applicant: {applicant_id}")

    # Basic check for expected key
    if "utility_payments" not in payload or not isinstance(payload["utility_payments"], list):
        logger.error(f"Invalid payload structure for applicant {applicant_id}. Missing 'utility_payments' list.")
        # In a real scenario, raise an error or return a specific status
        return

    records_data = payload["utility_payments"]
    validated_records: List[UtilityPaymentRecord] = []
    validation_errors = []

    for i, record_data in enumerate(records_data):
        try:
            # Validate each record against the Pydantic model
            validated_record = UtilityPaymentRecord(**record_data)
            validated_records.append(validated_record)
            # logger.debug(f"Validated record {i+1} for applicant {applicant_id}")
        except ValidationError as e:
            logger.warning(f"Validation failed for record {i+1} for applicant {applicant_id}: {e}")
            validation_errors.append({"record_index": i, "errors": e.errors()})
        except Exception as e:
            logger.error(f"Unexpected error processing record {i+1} for applicant {applicant_id}: {e}")
            validation_errors.append({"record_index": i, "errors": "Unexpected processing error"})

    if validation_errors:
        logger.warning(f"Found {len(validation_errors)} validation errors during utility payment ingestion for applicant {applicant_id}.")
        # Handle errors appropriately - e.g., store them, notify admin, return error response

    logger.info(f"Successfully validated {len(validated_records)} utility payment records for applicant {applicant_id}.")

    # --- Store Validated Records ---
    if validated_records:
        logger.info(f"Attempting to save {len(validated_records)} validated records to database for applicant {applicant_id}.")
        db = None # Initialize db session variable
        try:
            db = SessionLocal() # Create a new session for this background task
            created_records = create_utility_payments_bulk(
                db=db, applicant_id=applicant_id, payments_in=validated_records
            )
            logger.info(f"Successfully saved {len(created_records)} utility payment records to database for applicant {applicant_id}.")
            # --- Next Step ---
            # Potentially trigger further processing or feature engineering tasks using created_records IDs
            # ------------------
        except Exception as e:
            # CRUD function logs details, just log context here
            logger.error(f"Database error during utility payment bulk save for applicant {applicant_id}: {e}")
            # Depending on requirements, might need to update ingestion log status to 'error'
        finally:
            if db:
                db.close() # Ensure the session is closed
                logger.debug(f"Database session closed for applicant {applicant_id} utility payment ingestion.")
    else:
        logger.info(f"No valid utility payment records to save for applicant {applicant_id}.")


    logger.info(f"Completed utility payment ingestion process for applicant: {applicant_id}")
