import uuid
from sqlalchemy import Column, String, DateTime, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from src.common.db import Base

class ApplicantFeatures(Base):
    """
    SQLAlchemy ORM model for storing engineered features for each applicant.
    """
    __tablename__ = "applicant_features"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    applicant_id = Column(String, nullable=False, index=True, unique=True)
    features = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<ApplicantFeatures(id={self.id}, applicant_id='{self.applicant_id}')>"

# Optionally, add a unique index on applicant_id for fast upserts
Index("ix_applicant_features_applicant_id", ApplicantFeatures.applicant_id, unique=True)
