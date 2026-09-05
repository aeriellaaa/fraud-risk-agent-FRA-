"""
Generates a real threshold-vs-cost sweep for the main (credit_card_fraud_2026) pipeline,
saved so /metrics can serve it without re-running the sweep on every request.

This exists because the frontend's "Net cost by threshold" chart previously showed entirely
fabricated points -- this produces the real equivalent from the actual trained model and
held-out test set, using the same sourced Rs94/Rs34802 cost figures as everywhere else.

Run with: python -m scripts.generate_cost_curve
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

FEATURE_ORDER = [
    "amount_usd", "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "is_ai_generated_scam_attempt", "merchant_risk_score",
    "prior_disputes",
]
CATEGORICAL_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
COST_FP = 94
COST_FN = 34802


def main():
    df = pd.read_csv("data/credit_card_fraud_2026.csv")
    with open("app/ml_artifacts/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    X = df[FEATURE_ORDER].copy()
    for col in CATEGORICAL_COLS:
        X[col] = encoders[col].transform(X[col])
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    y = df["is_fraud"].values

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with open("app/ml_artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
    probs = model.predict_proba(X_test)[:, 1]

    # Sample thresholds across the real observed probability range for a readable chart --
    # not the full fine-grained sweep (that's in training_results.txt's single best point),
    # this is for VISUALIZING the shape of the cost curve.
    sample_thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.28, 0.5]
    curve = []
    for t in sample_thresholds:
        pred = (probs >= t).astype(int)
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        cost = fp * COST_FP + fn * COST_FN
        curve.append({"threshold": t, "cost_inr": cost, "fp": fp, "fn": fn})

    Path("app/ml_artifacts").mkdir(exist_ok=True)
    with open("app/ml_artifacts/cost_curve.json", "w") as f:
        json.dump({"curve": curve, "cost_fp": COST_FP, "cost_fn": COST_FN}, f, indent=2)

    print("Cost curve (real, sourced Rs94/Rs34802):")
    for point in curve:
        print(f"  threshold={point['threshold']:.2f}  cost=Rs{point['cost_inr']:,}  "
              f"(fp={point['fp']}, fn={point['fn']})")
    print("\nSaved to app/ml_artifacts/cost_curve.json")


if __name__ == "__main__":
    main()
