"""
Trains the Phase 2 fraud model on credit_card_fraud_2026.csv.

Run with: python scripts\train_model.py

Outputs:
  app/ml_artifacts/model.pkl     -- trained RandomForestClassifier
  app/ml_artifacts/encoders.pkl  -- LabelEncoders for categorical columns
  Prints held-out precision/recall/AUC + a cost-optimal threshold sweep.

Cost figures below (Rs94 FP / Rs34802 FN) are VERIFIED and sourced -- see
inline comments for citations. See scripts/calibration_and_bootstrap_analysis.py
for calibration testing (not adopted -- see that script) and bootstrap CI.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA_PATH = Path("data/credit_card_fraud_2026.csv")
ARTIFACT_DIR = Path("app/ml_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

FEATURE_ORDER = [
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

COST_FP = 94      # Manual review cost: ~15 min at avg. Indian fraud analyst hourly rate of Rs 377/hr (ERI SalaryExpert, salaryexpert.com/salary/job/fraud-analyst/india)
COST_FN = 34802    # Missed fraud cost: avg. value of card/internet banking fraud in India, FY22, from Lok Sabha data via Business Standard (business-standard.com). FY22 is the most recent card/internet-specific figure with a clean citation -- newer RBI releases report aggregate fraud dominated by loan/advances fraud, a different category.


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, fraud rate {df['is_fraud'].mean():.4f}")

    X = df[FEATURE_ORDER].copy()
    y = df["is_fraud"].values

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    for col in BOOLEAN_COLS:
        X[col] = X[col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, test fraud count: {y_test.sum()}")

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nHeld-out AUC: {auc:.4f}")
    print(f"Probability range on test set: min={proba.min():.4f}, max={proba.max():.4f}")

    print("\n--- Cost-optimal threshold sweep ---")
    best_cost, best_threshold, best_stats = None, None, None
    for t in np.arange(0.01, 0.51, 0.01):
        preds = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = fp * COST_FP + fn * COST_FN
        if best_cost is None or cost < best_cost:
            best_cost, best_threshold = cost, round(float(t), 3)
            best_stats = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    precision = best_stats["tp"] / (best_stats["tp"] + best_stats["fp"]) if (best_stats["tp"] + best_stats["fp"]) else 0
    recall = best_stats["tp"] / (best_stats["tp"] + best_stats["fn"]) if (best_stats["tp"] + best_stats["fn"]) else 0

    print(f"Best threshold: {best_threshold} | total cost: {best_cost} (FP=Rs{COST_FP}, FN=Rs{COST_FN}, sourced -- see script header)")
    print(f"Confusion matrix: {best_stats}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

    default_preds = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, default_preds).ravel()
    default_cost = fp * COST_FP + fn * COST_FN
    print(f"\nFor comparison, cost at default 0.5 threshold: {default_cost}")

    with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ARTIFACT_DIR / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    with open(ARTIFACT_DIR / "training_results.txt", "w") as f:
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Best threshold: {best_threshold}\n")
        f.write(f"Best cost: {best_cost}\n")
        f.write(f"Confusion matrix: {best_stats}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Default 0.5 threshold cost: {default_cost}\n")

    print(f"\nSaved model + encoders to {ARTIFACT_DIR}/")
    print("Saved training_results.txt -- paste its contents back so we can wire the real thresholds in.")


if __name__ == "__main__":
    main()

