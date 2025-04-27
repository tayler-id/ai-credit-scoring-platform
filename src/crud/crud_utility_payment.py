import logging
from typing import List
from sqlalchemy.orm import Session

# Import the ORM model and the Pydantic model
from src.models.utility_payment import UtilityPayment
from src.common.models import UtilityPaymentRecord

logger = logging.getLogger(__name__)

def create_utility_payment(db: Session, *, applicant_id: str, payment_in: UtilityPaymentRecord) -> UtilityPayment:
    """
    Creates a single utility payment record in the database.

    Args:
        db: SQLAlchemy database session.
        applicant_id: The ID of the applicant associated with this payment.
        payment_in: The Pydantic model containing the utility payment data.

    Returns:
        The created UtilityPayment ORM object.
    """
    db_obj = UtilityPayment(
        applicant_id=applicant_id,
        provider=payment_in.provider,
        account_number=payment_in.account_number,
        payment_date=payment_in.payment_date,
        due_date=payment_in.due_date,
        amount_paid=payment_in.amount_paid,
        currency=payment_in.currency,
        payment_status=payment_in.payment_status,
        bill_period_start=payment_in.bill_period_start,
        bill_period_end=payment_in.bill_period_end
        # id, created_at, updated_at are handled by defaults/DB
    )
    try:
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        logger.info(f"Created utility payment record with ID: {db_obj.id} for applicant: {applicant_id}")
        return db_obj
    except Exception as e:
        logger.error(f"Error creating utility payment record for applicant {applicant_id}: {e}", exc_info=True)
        db.rollback()
        raise

def create_utility_payments_bulk(db: Session, *, applicant_id: str, payments_in: List[UtilityPaymentRecord]) -> List[UtilityPayment]:
    """
    Creates multiple utility payment records in the database efficiently.

    Args:
        db: SQLAlchemy database session.
        applicant_id: The ID of the applicant associated with these payments.
        payments_in: A list of Pydantic models containing the utility payment data.

    Returns:
        A list of the created UtilityPayment ORM objects.
    """
    db_objs = [
        UtilityPayment(
            applicant_id=applicant_id,
            provider=p.provider,
            account_number=p.account_number,
            payment_date=p.payment_date,
            due_date=p.due_date,
            amount_paid=p.amount_paid,
            currency=p.currency,
            payment_status=p.payment_status,
            bill_period_start=p.bill_period_start,
            bill_period_end=p.bill_period_end
        ) for p in payments_in
    ]

    if not db_objs:
        return [] # Nothing to add

    try:
        db.add_all(db_objs) # More efficient for multiple objects
        db.commit()
        # Note: db.refresh() doesn't work directly with add_all like this.
        # The objects in db_objs will have their IDs populated after commit
        # if the primary key generation strategy allows (like UUID default).
        logger.info(f"Bulk created {len(db_objs)} utility payment records for applicant: {applicant_id}")
        # Return the list of objects which should now have IDs assigned by the DB commit
        return db_objs
    except Exception as e:
        logger.error(f"Error bulk creating utility payment records for applicant {applicant_id}: {e}", exc_info=True)
        db.rollback()
        raise

def get_utility_payments_by_applicant(db: Session, applicant_id: str, skip: int = 0, limit: int = 100) -> List[UtilityPayment]:
    """
    Retrieves utility payment records for a specific applicant with pagination.

    Args:
        db: SQLAlchemy database session.
        applicant_id: The applicant's identifier.
        skip: Number of records to skip (for pagination).
        limit: Maximum number of records to return (for pagination).

    Returns:
        A list of UtilityPayment ORM objects.
    """
    return db.query(UtilityPayment)\
             .filter(UtilityPayment.applicant_id == applicant_id)\
             .order_by(UtilityPayment.payment_date.desc())\
             .offset(skip)\
             .limit(limit)\
             .all()

# Add other CRUD functions as needed (e.g., get by ID, update, delete)
