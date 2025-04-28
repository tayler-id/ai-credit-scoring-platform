from datetime import date
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
    Response model containing the credit score and SHAP explanations.
    """
    applicant_id: str
    score: float | None = None
    risk_level: str | None = None # e.g., 'low', 'medium', 'high'
    status: str = Field(default="pending", description="Status of the scoring process (e.g., pending, completed, error)")
    message: str | None = None
    shap_explanation: dict | None = None  # SHAP values and explanation for the prediction

# Add more specific data models as needed, e.g.:
# class MobileMoneyTransaction(BaseModel): ...

class UtilityPaymentRecord(BaseModel):
    """Represents a single utility payment record."""
    provider: str = Field(..., description="Name of the utility provider (e.g., 'City Power', 'Water Board')")
    account_number: str | None = Field(None, description="Utility account number associated with the payment")
    payment_date: date = Field(..., description="Date the payment was made")
    due_date: date | None = Field(None, description="Original due date of the bill")
    amount_paid: float = Field(..., description="Amount paid")
    currency: str = Field(..., description="Currency of the payment (e.g., 'USD', 'KES')")
    payment_status: str = Field(..., description="Status of the payment (e.g., 'paid_on_time', 'paid_late', 'missed')")
    bill_period_start: date | None = Field(None, description="Start date of the billing period")
    bill_period_end: date | None = Field(None, description="End date of the billing period")

# class UtilityBill(BaseModel): ... # This might represent a full bill with multiple line items, potentially containing UtilityPaymentRecord(s)

# MVP Basic Data Models
class BasicProfileCreate(BaseModel):
    """
    Request model to create a basic profile for the MVP.
    """
    applicant_id: str = Field(..., description="Unique identifier for the applicant")
    name: str = Field(..., description="Applicant's name")
    phone_number: str = Field(..., description="Applicant's phone number")
    occupation: str | None = Field(default=None, description="Applicant's occupation or type of business")
    years_in_business: int | None = Field(default=None, description="Number of years in business")

class DeclaredIncomeCreate(BaseModel):
    """
    Request model to create declared income data for the MVP.
    """
    applicant_id: str = Field(..., description="Unique identifier for the applicant")
    monthly_income: float = Field(..., description="Applicant's self-declared monthly income")

class MVPScoreRequest(BaseModel):
    """
    Request model to get a basic MVP credit score for an applicant.
    """
    applicant_id: str = Field(..., description="Unique identifier for the applicant")

class MVPScoreResponse(BaseModel):
    """
    Response model containing the basic MVP credit score.
    """
    applicant_id: str
    basic_score: float | None = None
    status: str = Field(default="pending", description="Status of the scoring process (e.g., pending, completed, error)")
    message: str | None = None
