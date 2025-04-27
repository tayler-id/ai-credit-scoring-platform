"""
Mobile Money Data Ingestion Module

Defines the expected schema for mobile money transactions and provides a mock ingestion function.
This module is the foundation for feature engineering and integration into the credit scoring pipeline.
"""

from typing import List, Dict
import pandas as pd

# Example schema for a mobile money transaction record
MOBILE_MONEY_SCHEMA = [
    "transaction_id",      # Unique transaction identifier
    "applicant_id",        # Foreign key to applicant
    "timestamp",           # Datetime of transaction
    "amount",              # Transaction amount (float)
    "balance",             # Account balance after transaction (float)
    "transaction_type",    # e.g., 'cash_in', 'cash_out', 'p2p_send', 'p2p_receive', 'bill_pay', etc.
    "counterparty",        # Optional: phone number or merchant ID
    "channel",             # e.g., 'USSD', 'app', 'agent'
    "status"               # e.g., 'success', 'failed'
]

def mock_fetch_mobile_money_transactions(applicant_id: str) -> pd.DataFrame:
    """
    Mock function to fetch mobile money transactions for a given applicant.
    In production, this would connect to an API or data source.
    """
    data = [
        {
            "transaction_id": "txn1",
            "applicant_id": applicant_id,
            "timestamp": "2025-04-01T10:00:00",
            "amount": 50.0,
            "balance": 150.0,
            "transaction_type": "cash_in",
            "counterparty": "agent_123",
            "channel": "USSD",
            "status": "success"
        },
        {
            "transaction_id": "txn2",
            "applicant_id": applicant_id,
            "timestamp": "2025-04-02T12:30:00",
            "amount": -20.0,
            "balance": 130.0,
            "transaction_type": "bill_pay",
            "counterparty": "utility_456",
            "channel": "app",
            "status": "success"
        },
        {
            "transaction_id": "txn3",
            "applicant_id": applicant_id,
            "timestamp": "2025-04-03T09:15:00",
            "amount": -10.0,
            "balance": 120.0,
            "transaction_type": "p2p_send",
            "counterparty": "user_789",
            "channel": "USSD",
            "status": "success"
        },
        # Add more mock transactions as needed
    ]
    df = pd.DataFrame(data)
    return df
