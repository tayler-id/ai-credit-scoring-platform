import logging
from sqlalchemy.orm import Session
from src.models.ingestion_log import IngestionLog
from src.models.utility_payment import UtilityPayment
import pandas as pd
import numpy as np

# --- New imports for mobile money ---
from src.data_ingestion.mobile_money import mock_fetch_mobile_money_transactions
from src.data_processing.mobile_money_processor import engineer_mobile_money_features

# --- New imports for e-commerce ---
from src.data_ingestion.ecommerce import mock_fetch_ecommerce_transactions
from src.data_processing.ecommerce_processor import engineer_ecommerce_features

logger = logging.getLogger(__name__)

def process_data_for_applicant(db: Session, applicant_id: str):
    """
    Main function to orchestrate data processing for a given applicant.
    This would likely be triggered after data ingestion (e.g., by an async task).

    Args:
        db: SQLAlchemy database session.
        applicant_id: The identifier of the applicant whose data needs processing.
    """
    logger.info(f"Starting data processing for applicant: {applicant_id}")

    # 1. Fetch and process utility payment data
    raw_logs = fetch_raw_data(db, applicant_id)
    if not raw_logs:
        logger.warning(f"No raw data found to process for applicant: {applicant_id}")
        utility_features = {}
    else:
        cleaned_data = clean_transform_data(raw_logs)
        utility_features = engineer_features(cleaned_data)

    # 2. Fetch and process mobile money data
    mm_df = mock_fetch_mobile_money_transactions(applicant_id)
    mm_features = engineer_mobile_money_features(mm_df)

    # 3. Fetch and process e-commerce data
    ec_df = mock_fetch_ecommerce_transactions(applicant_id)
    ec_features = engineer_ecommerce_features(ec_df)

    # 4. Merge features
    features = {}
    features.update(utility_features)
    features.update(mm_features)
    features.update(ec_features)

    # 5. Save features to a Feature Store or database table
    save_features(db, applicant_id, features)

    # 6. Update status (e.g., mark ingestion logs as 'processed' or update applicant status)
    # Example: update_ingestion_status(db, raw_logs, 'processed')
    update_status_placeholder(db, applicant_id, 'processed')

    logger.info(f"Completed data processing for applicant: {applicant_id}")


# --- Placeholder Helper Functions ---

def fetch_raw_data(db: Session, applicant_id: str) -> list:
    """Fetches all utility payment records for the applicant."""
    logger.info(f"Fetching utility payment data for applicant {applicant_id}")
    payments = db.query(UtilityPayment).filter(UtilityPayment.applicant_id == applicant_id).all()
    if not payments:
        logger.warning(f"No utility payment records found for applicant {applicant_id}")
    return payments

def clean_transform_data(payments: list) -> pd.DataFrame:
    """Cleans and transforms utility payment data into a DataFrame."""
    logger.info("Cleaning and transforming utility payment data")
    if not payments:
        return pd.DataFrame()
    records = []
    for p in payments:
        records.append({
            "payment_date": p.payment_date,
            "due_date": p.due_date,
            "amount_paid": p.amount_paid,
            "payment_status": p.payment_status,
            "bill_period_start": p.bill_period_start,
            "bill_period_end": p.bill_period_end,
            "currency": p.currency,
            "provider": p.provider,
        })
    df = pd.DataFrame(records)
    # Handle missing values and add on_time/late flags
    df["on_time"] = np.where(
        (df["due_date"].notnull()) & (df["payment_date"] <= df["due_date"]),
        1, 0
    )
    df["late"] = np.where(
        (df["due_date"].notnull()) & (df["payment_date"] > df["due_date"]),
        1, 0
    )
    return df

def engineer_features(df: pd.DataFrame) -> dict:
    """Engineers features from cleaned utility payment DataFrame."""
    logger.info("Engineering features from utility payment data")
    if df.empty:
        return {}
    features = {}
    features["num_payments"] = len(df)
    features["avg_payment_amount"] = df["amount_paid"].mean()
    features["on_time_payment_rate"] = df["on_time"].mean()
    features["num_late_payments"] = df["late"].sum()
    features["num_missed_payments"] = (df["payment_status"] == "missed").sum()
    features["payment_history_months"] = (
        (df["payment_date"].max() - df["payment_date"].min()).days / 30.0
        if len(df) > 1 else 0
    )
    features["payment_frequency"] = (
        features["num_payments"] / features["payment_history_months"]
        if features["payment_history_months"] > 0 else features["num_payments"]
    )
    features["currency"] = df["currency"].mode()[0] if not df["currency"].empty else None
    features["provider"] = df["provider"].mode()[0] if not df["provider"].empty else None
    return features

from src.models.applicant_features import ApplicantFeatures

def save_features(db: Session, applicant_id: str, features: dict):
    """Upserts engineered features for the applicant into the database."""
    logger.info(f"Saving features for {applicant_id}: {features}")
    existing = db.query(ApplicantFeatures).filter(ApplicantFeatures.applicant_id == applicant_id).first()
    if existing:
        existing.features = features
    else:
        new_entry = ApplicantFeatures(applicant_id=applicant_id, features=features)
        db.add(new_entry)
    db.commit()

def update_status_placeholder(db: Session, applicant_id: str, status: str):
    """Simulates updating processing status."""
    logger.info(f"Simulating updating status to '{status}' for {applicant_id}")
    # Replace with actual status update logic

# ------------------------------------
