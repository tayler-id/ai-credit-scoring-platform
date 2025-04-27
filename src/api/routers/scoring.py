from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from src.common.models import CreditScoreRequest, CreditScoreResponse
from src.common.db import get_db
from src.ml_models.predictor import predict_score # Import the placeholder predictor
# from src.models.ingestion_log import IngestionLog # Might need this later
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/score",
    response_model=CreditScoreResponse,
    status_code=status.HTTP_200_OK, # Changed to 200 as it's now synchronous (placeholder)
    summary="Request Credit Score",
    description="Requests and returns a credit score calculation for a given applicant ID based on available data.",
    tags=["Scoring"]
)
async def request_credit_score(
    request: CreditScoreRequest,
    db: Session = Depends(get_db)
):
    """
    Endpoint to request a credit score for an applicant.

    - **applicant_id**: Unique identifier for the applicant.

    This placeholder implementation runs the scoring synchronously.
    A production system would likely:
    1. Check if sufficient data exists.
    2. Trigger an asynchronous scoring task (e.g., via Celery).
    3. Return a task ID or 'processing' status immediately (HTTP 202).
    4. Provide a separate endpoint to poll for the result.
    """
    logger.info(f"Received scoring request for applicant_id: {request.applicant_id}")

    # --- Placeholder: Check if applicant data exists ---
    # Example:
    # data_exists = check_data_readiness(db, request.applicant_id)
    # if not data_exists:
    #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Insufficient data found for applicant to generate score.")
    # -------------------------------------------------

    try:
        # Call the updated prediction function (now returns SHAP explanation)
        score, risk_level, shap_explanation = predict_score(db=db, applicant_id=request.applicant_id)

        if score is None or risk_level is None:
            logger.warning(f"Score prediction failed or returned None for applicant: {request.applicant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, # Or 500 if it's an internal error
                detail="Could not generate score for applicant. Required data might be missing or model error occurred."
            )

        status_msg = "Score calculated successfully."
        response_status = "completed"
        logger.info(f"Score generated for applicant {request.applicant_id}: Score={score}, Risk={risk_level}, SHAP={shap_explanation}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during scoring for applicant {request.applicant_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during the scoring process."
        )

    return CreditScoreResponse(
        applicant_id=request.applicant_id,
        score=score,
        risk_level=risk_level,
        status=response_status,
        message=status_msg,
        shap_explanation=shap_explanation
    )

# Example of a potential endpoint to retrieve the score later (if async)
# @router.get("/score/{applicant_id}", response_model=CreditScoreResponse, tags=["Scoring"])
# async def get_score_result(applicant_id: str, db: Session = Depends(get_db)):
#     # ... logic to fetch score ...
#     pass
