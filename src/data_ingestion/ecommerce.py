"""
E-Commerce Data Ingestion Module

Defines the expected schema for e-commerce transactions and provides a mock ingestion function.
This module is the foundation for feature engineering and integration into the credit scoring pipeline.
"""

from typing import List, Dict
import pandas as pd

# Example schema for an e-commerce transaction record
ECOMMERCE_SCHEMA = [
    "transaction_id",      # Unique transaction identifier
    "applicant_id",        # Foreign key to applicant
    "timestamp",           # Datetime of transaction
    "amount",              # Transaction amount (float)
    "merchant_id",         # Unique merchant identifier
    "merchant_category",   # e.g., 'electronics', 'groceries', 'fashion', etc.
    "payment_method",      # e.g., 'card', 'mobile_money', 'cash_on_delivery'
    "status",              # e.g., 'completed', 'cancelled', 'failed'
    "item_count"           # Number of items in the transaction
]

def mock_fetch_ecommerce_transactions(applicant_id: str) -> pd.DataFrame:
    """
    Mock function to fetch e-commerce transactions for a given applicant.
    In production, this would connect to an API or data source.
    """
    data = [
        {
            "transaction_id": "ec_txn1",
            "applicant_id": applicant_id,
            "timestamp": "2025-04-01T14:20:00",
            "amount": 120.0,
            "merchant_id": "m_001",
            "merchant_category": "electronics",
            "payment_method": "card",
            "status": "completed",
            "item_count": 2
        },
        {
            "transaction_id": "ec_txn2",
            "applicant_id": applicant_id,
            "timestamp": "2025-04-05T09:45:00",
            "amount": 35.5,
            "merchant_id": "m_002",
            "merchant_category": "groceries",
            "payment_method": "mobile_money",
            "status": "completed",
            "item_count": 10
        },
        {
            "transaction_id": "ec_txn3",
            "applicant_id": applicant_id,
            "timestamp": "2025-04-10T18:10:00",
            "amount": 60.0,
            "merchant_id": "m_003",
            "merchant_category": "fashion",
            "payment_method": "card",
            "status": "completed",
            "item_count": 1
        },
        # Add more mock transactions as needed
    ]
    df = pd.DataFrame(data)
    return df
