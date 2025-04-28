import os
import sys
import uuid
import random
from datetime import date, timedelta
from faker import Faker
from sqlalchemy.orm import Session

# Add project root to Python path
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

from src.common.db import SessionLocal, engine, Base
from src.models.utility_payment import UtilityPayment # Corrected import name
from src.models.supply_chain import SupplyChainTransaction
# Import other raw data models as needed

# Initialize Faker
fake = Faker()

# --- Configuration ---
NUM_APPLICANTS = 20
RECORDS_PER_APPLICANT = 50
START_DATE = date(2023, 1, 1)
END_DATE = date(2024, 12, 31)
UTILITY_PROVIDERS = ["City Power", "Water Board", "Gas Co", "Telecom Plus"]
PAYMENT_STATUSES = ["paid_on_time", "paid_late", "missed"]
SUPPLIERS = [fake.company() for _ in range(10)]
BUYERS = [fake.company() for _ in range(10)]
TRANSACTION_TYPES = ["invoice", "payment", "credit_note"]
CURRENCY = "USD" # Assuming a single currency for simplicity

def generate_utility_payments(db: Session, applicant_id: str):
    """Generates synthetic utility payment records for one applicant."""
    records = []
    for _ in range(RECORDS_PER_APPLICANT):
        payment_date = fake.date_between(start_date=START_DATE, end_date=END_DATE)
        due_date_delta = timedelta(days=random.randint(-10, 10)) # Due date around payment date
        due_date = payment_date + due_date_delta
        bill_period_end = payment_date - timedelta(days=random.randint(1, 5)) # Bill period ends shortly before payment
        bill_period_start = bill_period_end - timedelta(days=random.randint(28, 35)) # Approx 1 month bill period

        record = UtilityPayment( # Corrected class name
            applicant_id=applicant_id,
            provider=random.choice(UTILITY_PROVIDERS),
            account_number=fake.iban(), # Using IBAN as placeholder account number
            payment_date=payment_date,
            due_date=due_date,
            amount_paid=round(random.uniform(20.0, 200.0), 2),
            currency=CURRENCY,
            payment_status=random.choices(PAYMENT_STATUSES, weights=[0.8, 0.15, 0.05], k=1)[0], # Weighted choices
            bill_period_start=bill_period_start,
            bill_period_end=bill_period_end
        )
        records.append(record)
    db.add_all(records)

def generate_supply_chain_transactions(db: Session, applicant_id: str):
    """Generates synthetic supply chain transaction records for one applicant."""
    records = []
    # Assume the applicant is sometimes the buyer, sometimes the supplier
    for _ in range(RECORDS_PER_APPLICANT):
        transaction_date = fake.date_between(start_date=START_DATE, end_date=END_DATE)
        is_buyer = random.choice([True, False])
        partner_id = random.choice(BUYERS) if is_buyer else random.choice(SUPPLIERS)
        partner_type = "customer" if is_buyer else "supplier"
        transaction_type = random.choice(TRANSACTION_TYPES)
        status = random.choice(["pending", "completed", "disputed"])
        unit_price = round(random.uniform(5.0, 500.0), 2)
        quantity = random.randint(1, 100)
        total_amount = unit_price * quantity

        record = SupplyChainTransaction(
            applicant_id=applicant_id,
            transaction_date=transaction_date,
            partner_id=partner_id,
            partner_type=partner_type,
            transaction_type=transaction_type,
            amount=total_amount,
            currency=CURRENCY,
            status=status,
            description=fake.bs()
            # Removed fields not present in the actual model:
            # transaction_id, supplier_id, buyer_id, product_description,
            # quantity, unit_price, payment_terms
        )
        records.append(record)
    db.add_all(records)


def main():
    print("Generating synthetic data...")
    # Ensure tables exist (using synchronous engine for script)
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        applicant_ids = [f"synthetic_{i+1:03d}" for i in range(NUM_APPLICANTS)]

        for i, app_id in enumerate(applicant_ids):
            print(f"Generating data for applicant {i+1}/{NUM_APPLICANTS}: {app_id}")
            generate_utility_payments(db, app_id)
            generate_supply_chain_transactions(db, app_id)
            # Add calls to generate other data types here

        print("Committing data to database...")
        db.commit()
        print("Synthetic data generation complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
