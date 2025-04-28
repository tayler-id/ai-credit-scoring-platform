from sqlalchemy.orm import Session
from src.models.mvp_basic_data import BasicProfile, DeclaredIncome
from src.common.db import Base # Although Base is not directly used here, it's common to import it in CRUD files

def create_basic_profile(db: Session, applicant_id: str, name: str, phone_number: str, occupation: str = None, years_in_business: int = None):
    """
    Creates a new basic profile record.
    """
    db_profile = BasicProfile(
        applicant_id=applicant_id,
        name=name,
        phone_number=phone_number,
        occupation=occupation,
        years_in_business=years_in_business
    )
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile

def get_basic_profile_by_applicant_id(db: Session, applicant_id: str):
    """
    Retrieves a basic profile record by applicant ID.
    """
    return db.query(BasicProfile).filter(BasicProfile.applicant_id == applicant_id).first()

def create_declared_income(db: Session, applicant_id: str, monthly_income: float):
    """
    Creates a new declared income record.
    """
    db_income = DeclaredIncome(
        applicant_id=applicant_id,
        monthly_income=monthly_income
    )
    db.add(db_income)
    db.commit()
    db.refresh(db_income)
    return db_income

def get_declared_income_by_applicant_id(db: Session, applicant_id: str):
    """
    Retrieves a declared income record by applicant ID.
    """
    return db.query(DeclaredIncome).filter(DeclaredIncome.applicant_id == applicant_id).first()

# Note: Update functions can be added later if needed for the MVP.
