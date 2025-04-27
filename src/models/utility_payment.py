import uuid
from sqlalchemy import Column, String, DateTime, Date, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.common.db import Base  # Import the Base from our db setup

class UtilityPayment(Base):
    """
    SQLAlchemy ORM model for storing utility payment records.
    Represents the 'utility_payments' table in the database.
    """
    __tablename__ = "utility_payments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign key to the applicant - assuming an 'applicants' table might exist later
    # For now, just storing the ID provided in the payload. Add index for faster lookups.
    applicant_id = Column(String, nullable=False, index=True)

    # Consider adding a foreign key to ingestion_log if needed for traceability
    # ingestion_log_id = Column(UUID(as_uuid=True), ForeignKey("ingestion_logs.id"))

    # Fields corresponding to the UtilityPaymentRecord Pydantic model
    provider = Column(String, nullable=False)
    account_number = Column(String, nullable=True)
    payment_date = Column(Date, nullable=False)
    due_date = Column(Date, nullable=True)
    amount_paid = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False) # Assuming 3-char currency codes
    payment_status = Column(String, nullable=False, index=True) # Index common query field
    bill_period_start = Column(Date, nullable=True)
    bill_period_end = Column(Date, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<UtilityPayment(id={self.id}, applicant_id='{self.applicant_id}', provider='{self.provider}', date='{self.payment_date}', amount={self.amount_paid})>"

# Note: Remember to use a migration tool like Alembic to create/update the table schema.
