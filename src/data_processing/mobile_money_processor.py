"""
Mobile Money Feature Engineering Module

Processes mobile money transaction data and engineers features for credit scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def engineer_mobile_money_features(transactions: pd.DataFrame) -> Dict[str, Any]:
    """
    Given a DataFrame of mobile money transactions for a single applicant,
    engineer features relevant for credit scoring.
    """
    features = {}

    if transactions.empty:
        # Return default/zero features if no data
        features = {
            "mm_num_transactions": 0,
            "mm_total_inflow": 0.0,
            "mm_total_outflow": 0.0,
            "mm_avg_balance": 0.0,
            "mm_cash_in_ratio": 0.0,
            "mm_cash_out_ratio": 0.0,
            "mm_p2p_send_count": 0,
            "mm_p2p_receive_count": 0,
            "mm_bill_pay_count": 0,
            "mm_active_days": 0,
            "mm_avg_txn_amount": 0.0,
            "mm_payment_regular_days": 0
        }
        return features

    # Ensure timestamp is datetime
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])

    # Basic counts and sums
    features["mm_num_transactions"] = len(transactions)
    features["mm_total_inflow"] = transactions[transactions["amount"] > 0]["amount"].sum()
    features["mm_total_outflow"] = -transactions[transactions["amount"] < 0]["amount"].sum()
    features["mm_avg_balance"] = transactions["balance"].mean() if not transactions["balance"].empty else 0.0

    # Transaction type ratios
    total = len(transactions)
    features["mm_cash_in_ratio"] = (
        len(transactions[transactions["transaction_type"] == "cash_in"]) / total if total else 0.0
    )
    features["mm_cash_out_ratio"] = (
        len(transactions[transactions["transaction_type"] == "cash_out"]) / total if total else 0.0
    )

    # P2P and bill pay
    features["mm_p2p_send_count"] = len(transactions[transactions["transaction_type"] == "p2p_send"])
    features["mm_p2p_receive_count"] = len(transactions[transactions["transaction_type"] == "p2p_receive"])
    features["mm_bill_pay_count"] = len(transactions[transactions["transaction_type"] == "bill_pay"])

    # Activity metrics
    features["mm_active_days"] = transactions["timestamp"].dt.date.nunique()
    features["mm_avg_txn_amount"] = transactions["amount"].mean() if not transactions["amount"].empty else 0.0

    # Payment regularity: number of days with at least one transaction
    features["mm_payment_regular_days"] = features["mm_active_days"]

    return features
