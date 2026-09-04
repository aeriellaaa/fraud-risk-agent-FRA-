"""
Capacity Cross-Check: does the cost-optimal threshold actually respect any realistic
review-team's capacity? Run with: python -m scripts.capacity_cross_check
"""
import pandas as pd
import numpy as np
import pickle
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from app.capacity import compare_cost_optimal_vs_capacity_constrained

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
COST_OPTIMAL_THRESHOLD = 0.01


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

    result = compare_cost_optimal_vs_capacity_constrained(
        probs=probs, y_true=y_test, cost_optimal_threshold=COST_OPTIMAL_THRESHOLD,
        team_sizes=[1, 2, 5, 10, 20], daily_transaction_volume=5000,
    )
    print(f"Assumed daily transaction volume: {result['daily_transaction_volume']:,}\n")
    print(f"{'Scenario':<38} {'Threshold':>10} {'Escalate%':>10} {'Recall':>8} {'Sustainable':>12}")
    print("-" * 82)
    for row in result["rows"]:
        thresh = f"{row['auto_approve_threshold']:.4f}" if row["auto_approve_threshold"] is not None else "N/A"
        esc = f"{row['escalate_rate']:.2%}" if row["escalate_rate"] is not None else "N/A"
        rec = f"{row['recall']:.1%}" if row["recall"] is not None else "N/A"
        print(f"{row['scenario']:<38} {thresh:>10} {esc:>10} {rec:>8} {str(row['sustainable']):>12}")

    Path("app/ml_artifacts").mkdir(exist_ok=True)
    with open("app/ml_artifacts/capacity_cross_check_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved results to app/ml_artifacts/capacity_cross_check_results.json")


if __name__ == "__main__":
    main()
