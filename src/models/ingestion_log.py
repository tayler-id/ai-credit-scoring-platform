import uuid
from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.common.db import Base  # Import the Base from our db setup

class IngestionLog(Base):
    """
    SQLAlchemy ORM model for storing ingestion event logs.
    Represents the 'ingestion_logs' table in the database.
    """
    __tablename__ = "ingestion_logs"

    # Use the same UUID generated during the ingestion request as the primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    applicant_id = Column(String, nullable=False, index=True)
    data_source = Column(String, nullable=False, index=True)

    # Store the raw payload as JSON(B) for flexibility
    # Use JSONB if using PostgreSQL for better indexing capabilities
    payload = Column(JSON, nullable=False)

    status = Column(String, nullable=False, default="received", index=True) # e.g., received, processing, processed, error

    received_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<IngestionLog(id={self.id}, applicant_id='{self.applicant_id}', source='{self.data_source}', status='{self.status}')>"

# Note: To create this table in the database, you would typically use a migration tool
# like Alembic. You would run `alembic revision --autogenerate -m "Create ingestion_logs table"`
# and then `alembic upgrade head`.
# For initial setup/testing without Alembic, you could manually create the table
# or use `Base.metadata.create_all(bind=engine)` from `src.common.db` (less ideal for production).
