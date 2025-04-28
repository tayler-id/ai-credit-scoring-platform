import logging
from sqlalchemy.orm import Session
from src.models.ingestion_log import IngestionLog
from src.models.utility_payment import UtilityPayment
from src.models.supply_chain import SupplyChainTransaction # Import model
# Import other models as needed (MobileMoney, ECommerce)
import pandas as pd
import numpy as np

# --- Processor function imports ---
# Utility features are engineered within this file's helpers
from src.data_processing.mobile_money_processor import engineer_mobile_money_features # Assuming this exists
from src.data_processing.ecommerce_processor import engineer_ecommerce_features # Assuming this exists
from src.data_processing.supply_chain_processor import engineer_supply_chain_features # Use the refactored version

# --- CRUD function imports ---
# from src.crud.crud_mobile_money import get_mobile_money_transactions_by_applicant # Placeholder
# from src.crud.crud_ecommerce import get_ecommerce_transactions_by_applicant # Placeholder
from src.crud.crud_supply_chain import get_supply_chain_transactions_by_applicant # Now implemented


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
    # TODO: Replace mock fetch with actual DB query using CRUD
    # mm_transactions = get_mobile_money_transactions_by_applicant(db, applicant_id)
    # mm_features = engineer_mobile_money_features(mm_transactions) # Assuming processor takes list of objects
    logger.warning(f"Mobile money data fetching not implemented for applicant {applicant_id}. Using empty features.")
    mm_features = {} # Placeholder

    # 3. Fetch and process e-commerce data
    # TODO: Replace mock fetch with actual DB query using CRUD
    # ec_transactions = get_ecommerce_transactions_by_applicant(db, applicant_id)
    # ec_features = engineer_ecommerce_features(ec_transactions) # Assuming processor takes list of objects
    logger.warning(f"E-commerce data fetching not implemented for applicant {applicant_id}. Using empty features.")
    ec_features = {} # Placeholder

    # 4. Fetch and process supply chain data
    # Fetch from DB using CRUD
    try:
        # Assuming a CRUD function exists or will be added to crud_supply_chain.py
        sc_transactions = get_supply_chain_transactions_by_applicant(db, applicant_id)
        sc_features = engineer_supply_chain_features(sc_transactions) # Pass list of objects
    except NameError: # Handle if CRUD function doesn't exist yet
         logger.error("CRUD function 'get_supply_chain_transactions_by_applicant' not found. Skipping supply chain features.")
         sc_features = {}
    except Exception as e:
        logger.error(f"Error processing supply chain data for applicant {applicant_id}: {e}", exc_info=True)
        sc_features = {}


    # 5. Merge features from all sources
    features = {}
    features.update(utility_features)
    features.update(mm_features)
    features.update(ec_features)
    features.update(sc_features) # Add supply chain features

    # 6. Save features to a Feature Store or database table
    save_features(db, applicant_id, features)

    # 7. Update status (e.g., mark ingestion logs as 'processed' or update applicant status)
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
import json

# Helper function to convert numpy types to standard Python types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle potential NaN/Inf values before converting to float
        if np.isnan(obj):
            return None # Or another representation like 'NaN' string
        elif np.isinf(obj):
            return None # Or another representation like 'Infinity' string
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

def save_features(db: Session, applicant_id: str, features: dict):
    """Upserts engineered features for the applicant into the database."""
    # Convert numpy types in the features dictionary before saving
    serializable_features = convert_numpy_types(features)
    logger.info(f"Saving serializable features for {applicant_id}: {serializable_features}")

    existing = db.query(ApplicantFeatures).filter(ApplicantFeatures.applicant_id == applicant_id).first()
    if existing:
        existing.features = serializable_features
    else:
        new_entry = ApplicantFeatures(applicant_id=applicant_id, features=serializable_features)
        db.add(new_entry)
    db.commit()

def update_status_placeholder(db: Session, applicant_id: str, status: str):
    """Simulates updating processing status."""
    logger.info(f"Simulating updating status to '{status}' for {applicant_id}")
    # Replace with actual status update logic

# ------------------------------------

if __name__ == "__main__":
    # This block allows running the processor directly as a script
    # For example, to process data for all synthetic applicants
    from src.common.db import SessionLocal

    logger.info("Starting standalone data processing run...")
    db = SessionLocal()
    try:
        # Fetch applicant IDs (e.g., from utility payments or a dedicated applicant table)
        # For now, let's assume we know the synthetic IDs
        # TODO: Fetch this dynamically in a real scenario
        applicant_ids = [f"synthetic_{i+1:03d}" for i in range(20)] # Match NUM_APPLICANTS in generation script

        if not applicant_ids:
            logger.warning("No applicant IDs found to process.")
        else:
            logger.info(f"Found {len(applicant_ids)} applicants to process.")
            for applicant_id in applicant_ids:
                process_data_for_applicant(db=db, applicant_id=applicant_id)
            logger.info("Standalone data processing run finished.")

    except Exception as e:
        logger.error(f"Error during standalone processing run: {e}", exc_info=True)
    finally:
        db.close()
