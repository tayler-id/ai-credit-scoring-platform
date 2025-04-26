from pydantic import BaseModel, Field
from typing import Dict, Any
import uuid

class BaseDataModel(BaseModel):
    """Base model with common fields like ID."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)

class AlternativeDataPayload(BaseModel):
    """
    Represents a generic payload of alternative data for a specific applicant.
    The structure of 'data' can be flexible initially.
    """
    applicant_id: str = Field(..., description="Unique identifier for the applicant")
    data_source: str = Field(..., description="Identifier for the source of the data (e.g., 'mobile_money_api', 'utility_bill_upload')")
    payload: Dict[str, Any] = Field(..., description="The actual alternative data payload")

class IngestionResponse(BaseModel):
    """
    Response model after data ingestion.
    """
    ingestion_id: uuid.UUID = Field(..., description="Unique ID for this ingestion event")
    status: str = Field(default="received", description="Status of the ingestion process")
    message: str | None = None

class CreditScoreRequest(BaseModel):
    """
    Request model to get a credit score for an applicant.
    """
    applicant_id: str = Field(..., description="Unique identifier for the applicant")

class CreditScoreResponse(BaseModel):
    """
    Response model containing the credit score.
    """
    applicant_id: str
    score: float | None = None
    risk_level: str | None = None # e.g., 'low', 'medium', 'high'
    status: str = Field(default="pending", description="Status of the scoring process (e.g., pending, completed, error)")
    message: str | None = None

# Add more specific data models as needed, e.g.:
# class MobileMoneyTransaction(BaseModel): ...
# class UtilityBill(BaseModel): ...
