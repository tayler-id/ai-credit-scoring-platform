from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from src.common.models import BasicProfileCreate, DeclaredIncomeCreate, MVPScoreRequest, MVPScoreResponse
from src.common.db import get_db
from src.crud import crud_mvp_basic_data
from src.ml_models.mvp_scorer import calculate_basic_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/basic_profile",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Basic Profile Data (MVP)",
    description="Receives basic profile data for an applicant for the MVP.",
    tags=["MVP"]
)
async def ingest_basic_profile(
    payload: BasicProfileCreate,
    db: Session = Depends(get_db)
):
    """
    Endpoint to ingest basic profile data for the MVP.

    - **applicant_id**: Unique identifier for the applicant.
    - **name**: Applicant's name.
    - **phone_number**: Applicant's phone number.
    - **occupation**: Applicant's occupation or type of business (optional).
    - **years_in_business**: Number of years in business (optional).
    """
    logger.info(f"Received basic profile ingestion request for applicant: {payload.applicant_id}")

    # Check if a profile already exists for this applicant
    existing_profile = crud_mvp_basic_data.get_basic_profile_by_applicant_id(db, applicant_id=payload.applicant_id)
    if existing_profile:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Basic profile already exists for applicant_id: {payload.applicant_id}"
        )

    try:
        db_profile = crud_mvp_basic_data.create_basic_profile(
            db=db,
            applicant_id=payload.applicant_id,
            name=payload.name,
            phone_number=payload.phone_number,
            occupation=payload.occupation,
            years_in_business=payload.years_in_business
        )
        logger.info(f"Successfully saved basic profile for applicant: {payload.applicant_id}")
        return {"message": "Basic profile data ingested successfully", "applicant_id": db_profile.applicant_id}
    except Exception as e:
        logger.error(f"Failed to ingest basic profile for applicant {payload.applicant_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest basic profile data."
        )

@router.post(
    "/declared_income",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Declared Income Data (MVP)",
    description="Receives declared income data for an applicant for the MVP.",
    tags=["MVP"]
)
async def ingest_declared_income(
    payload: DeclaredIncomeCreate,
    db: Session = Depends(get_db)
):
    """
    Endpoint to ingest declared income data for the MVP.

    - **applicant_id**: Unique identifier for the applicant.
    - **monthly_income**: Applicant's self-declared monthly income.
    """
    logger.info(f"Received declared income ingestion request for applicant: {payload.applicant_id}")

    # Check if declared income already exists for this applicant
    existing_income = crud_mvp_basic_data.get_declared_income_by_applicant_id(db, applicant_id=payload.applicant_id)
    if existing_income:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Declared income already exists for applicant_id: {payload.applicant_id}"
        )

    try:
        db_income = crud_mvp_basic_data.create_declared_income(
            db=db,
            applicant_id=payload.applicant_id,
            monthly_income=payload.monthly_income
        )
        logger.info(f"Successfully saved declared income for applicant: {payload.applicant_id}")
        return {"message": "Declared income data ingested successfully", "applicant_id": db_income.applicant_id}
    except Exception as e:
        logger.error(f"Failed to ingest declared income for applicant {payload.applicant_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest declared income data."
        )

@router.get( # Changed from POST to GET
    "/score",
    response_model=MVPScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Request Basic MVP Credit Score",
    description="Requests and returns a basic MVP credit score for a given applicant ID based on basic profile and declared income data.",
    tags=["MVP"]
)
async def request_mvp_basic_score(
    applicant_id: str, # Changed from request: MVPScoreRequest to applicant_id: str
    db: Session = Depends(get_db)
):
    """
    Endpoint to request a basic MVP credit score for an applicant.

    - **applicant_id**: Unique identifier for the applicant (passed as query parameter).
    """
    logger.info(f"Received basic MVP scoring request for applicant_id: {applicant_id}")

    # Retrieve basic profile and declared income data
    basic_profile = crud_mvp_basic_data.get_basic_profile_by_applicant_id(db, applicant_id=applicant_id)
    declared_income = crud_mvp_basic_data.get_declared_income_by_applicant_id(db, applicant_id=applicant_id)

    if not basic_profile or not declared_income:
        detail_msg = "Missing basic profile or declared income data for applicant."
        logger.warning(f"Basic MVP scoring failed for applicant {applicant_id}: {detail_msg}")
        # Raise HTTPException instead of returning a response directly for GET
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail_msg
        )

    try:
        # Calculate the basic MVP score
        basic_score = calculate_basic_score(profile=basic_profile, income=declared_income)

        status_msg = "Basic MVP score calculated successfully."
        response_status = "completed"
        logger.info(f"Basic MVP score generated for applicant {applicant_id}: Score={basic_score}")

    except Exception as e:
        logger.error(f"Unexpected error during basic MVP scoring for applicant {applicant_id}: {e}", exc_info=True)
        # Raise HTTPException for internal errors as well
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during the basic MVP scoring process."
        )

    return MVPScoreResponse(
        applicant_id=applicant_id,
        basic_score=basic_score,
        status=response_status,
        message=status_msg
    )
