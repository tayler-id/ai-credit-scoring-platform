"""
E-Commerce Feature Engineering Module

Processes e-commerce transaction data and engineers features for credit scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def engineer_ecommerce_features(transactions: pd.DataFrame) -> Dict[str, Any]:
    """
    Given a DataFrame of e-commerce transactions for a single applicant,
    engineer features relevant for credit scoring.
    """
    features = {}

    if transactions.empty:
        # Return default/zero features if no data
        features = {
            "ec_num_transactions": 0,
            "ec_total_spend": 0.0,
            "ec_avg_order_value": 0.0,
            "ec_purchase_frequency": 0.0,
            "ec_merchant_diversity": 0,
            "ec_avg_item_count": 0.0,
            "ec_active_days": 0,
            "ec_recency_days": None,
            "ec_completed_ratio": 0.0,
            "ec_category_diversity": 0
        }
        return features

    # Ensure timestamp is datetime
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])

    # Only consider completed transactions for spend/order value
    completed = transactions[transactions["status"] == "completed"]

    # Basic counts and sums
    features["ec_num_transactions"] = len(transactions)
    features["ec_total_spend"] = completed["amount"].sum()
    features["ec_avg_order_value"] = completed["amount"].mean() if not completed["amount"].empty else 0.0

    # Purchase frequency: transactions per unique day
    features["ec_active_days"] = transactions["timestamp"].dt.date.nunique()
    features["ec_purchase_frequency"] = (
        features["ec_num_transactions"] / features["ec_active_days"] if features["ec_active_days"] else 0.0
    )

    # Merchant and category diversity
    features["ec_merchant_diversity"] = transactions["merchant_id"].nunique()
    features["ec_category_diversity"] = transactions["merchant_category"].nunique()

    # Average item count per transaction
    features["ec_avg_item_count"] = transactions["item_count"].mean() if not transactions["item_count"].empty else 0.0

    # Recency: days since last transaction
    most_recent = transactions["timestamp"].max()
    features["ec_recency_days"] = (pd.Timestamp.now() - most_recent).days if not transactions["timestamp"].empty else None

    # Completed transaction ratio
    features["ec_completed_ratio"] = (
        len(completed) / len(transactions) if len(transactions) else 0.0
    )

    return features
