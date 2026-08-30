"""
Compares the trained ML model against two baselines, on the SAME
held-out test split (random_state=42, matching scripts/train_model.py)
and the SAME real, cited cost model (Rs94 FP / Rs34802 FN).

Baselines:
  1. Rules engine -- a hand-weighted scorer using the same signal logic
     as the original Phase 1 design (velocity, merchant risk, CVV
     retries, VPN, foreign transaction), independent of the trained
     model. This is what "a human wrote some if-statements" looks like.
  2. Naive (amount alone) -- score = normalized transaction amount.
     Same shape as kavach's "transaction amount alone" baseline.

Run with: python -m scripts.baseline_comparison
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

DATA_PATH = "data/credit_card_fraud_2026.csv"
COST_FP = 94
COST_FN = 34802

FEATURE_COLS = [
    "amount_usd", "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age", "account_balance_usd",
    "is_new_merchant", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
    "cvv_retry_count", "velocity_score", "time_of_day_hour", "day_of_week",
    "is_ai_generated_scam_attempt", "merchant_risk_score", "prior_disputes",
]
CATEGORICAL_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOLEAN_COLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
                "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]


def rules_baseline_score(row) -> float:
    """Independent hand-weighted rule scorer -- same signal logic as the
    original Phase 1 design, not derived from the trained model."""
    score = 0.0
    if row["velocity_score"] > 27.6:
        score += 0.25
    if row["merchant_risk_score"] > 47.3:
        score += 0.25
    if row["cvv_retry_count"] > 0:
        score += 0.20
    if row["used_vpn"]:
        score += 0.15
    if row["is_foreign_transaction"]:
        score += 0.15
    return min(1.0, score)


def naive_baseline_score(amounts: pd.Series) -> np.ndarray:
    """Score = normalized transaction amount. No fraud logic at all."""
    return (amounts / amounts.max()).values


def cost_optimal(y_true, scores, cost_fp=COST_FP, cost_fn=COST_FN):
    best_cost, best_threshold, best_precision, best_recall = None, None, None, None
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if best_cost is None or cost < best_cost:
            best_cost, best_threshold = cost, round(float(t), 3)
            best_precision = tp / (tp + fp) if (tp + fp) else 0
            best_recall = tp / (tp + fn) if (tp + fn) else 0
    return best_threshold, best_precision, best_recall, best_cost


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].copy()
    y = df["is_fraud"].values

    with open("app/ml_artifacts/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("app/ml_artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)

    X_encoded = X.copy()
    for col in CATEGORICAL_COLS:
        X_encoded[col] = encoders[col].transform(X_encoded[col])
    for col in BOOLEAN_COLS:
        X_encoded[col] = X_encoded[col].astype(int)

    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X_encoded, y, X, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Test set: {len(X_test)} transactions, {y_test.sum()} fraud\n")

    results = {}

    # 1. Rules baseline
    rules_scores = raw_test.apply(rules_baseline_score, axis=1).values
    results["Rules engine (hand-weighted)"] = {
        "roc_auc": roc_auc_score(y_test, rules_scores),
        "pr_auc": average_precision_score(y_test, rules_scores),
    }

    # 2. Naive baseline
    naive_scores = naive_baseline_score(raw_test["amount_usd"])
    results["Naive (amount alone)"] = {
        "roc_auc": roc_auc_score(y_test, naive_scores),
        "pr_auc": average_precision_score(y_test, naive_scores),
    }

    # 3. Trained ML model
    ml_scores = model.predict_proba(X_test)[:, 1]
    results["Random Forest (trained)"] = {
        "roc_auc": roc_auc_score(y_test, ml_scores),
        "pr_auc": average_precision_score(y_test, ml_scores),
    }

    all_scores = {"Rules engine (hand-weighted)": rules_scores,
                  "Naive (amount alone)": naive_scores,
                  "Random Forest (trained)": ml_scores}

    print(f"{'Approach':<32} {'ROC-AUC':>10} {'PR-AUC':>10} {'Cost-opt threshold':>20} {'Precision':>10} {'Recall':>10} {'Total cost (Rs)':>16}")
    print("-" * 112)
    for name, scores in all_scores.items():
        threshold, precision, recall, cost = cost_optimal(y_test, scores)
        print(f"{name:<32} {results[name]['roc_auc']:>10.4f} {results[name]['pr_auc']:>10.4f} "
              f"{threshold:>20.3f} {precision:>10.4f} {recall:>10.4f} {cost:>16,.0f}")

    print(f"\nBase rate (fraud prevalence in test set): {y_test.mean():.4f}")


if __name__ == "__main__":
    main()
