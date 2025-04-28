import uuid
from sqlalchemy import Column, String, Float, Integer, TypeDecorator
# from sqlalchemy.dialects.postgresql import UUID # Remove PostgreSQL specific import
from sqlalchemy.types import CHAR # Import CHAR for UUID storage in SQLite
from src.common.db import Base

# Custom Type for UUID storage across DBs
class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import UUID
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                # hexstring
                return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value

class BasicProfile(Base):
    """
    SQLAlchemy ORM model for storing basic profile data collected in the MVP.
    """
    __tablename__ = "mvp_basic_profiles"

    id = Column(GUID, primary_key=True, default=uuid.uuid4) # Use custom GUID type
    applicant_id = Column(String, nullable=False, unique=True) # Link to applicant
    name = Column(String, nullable=False)
    phone_number = Column(String, nullable=False, unique=True)
    occupation = Column(String, nullable=True)
    years_in_business = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<BasicProfile(id={self.id}, applicant_id='{self.applicant_id}')>"

class DeclaredIncome(Base):
    """
    SQLAlchemy ORM model for storing declared income data collected in the MVP.
    """
    __tablename__ = "mvp_declared_income"

    id = Column(GUID, primary_key=True, default=uuid.uuid4) # Use custom GUID type
    applicant_id = Column(String, nullable=False, unique=True) # Link to applicant
    monthly_income = Column(Float, nullable=False)

    def __repr__(self):
        return f"<DeclaredIncome(id={self.id}, applicant_id='{self.applicant_id}')>"
