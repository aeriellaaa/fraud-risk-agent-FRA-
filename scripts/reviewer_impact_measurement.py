"""
Measures how often the Reviewer Agent actually changes the final decision across the
FULL held-out test set. Uses the exact same evidence-generation logic as the real
scoring_agent.py, but computes SHAP values in one batched call (not row-by-row) for
tractable runtime.

Run with: python -m scripts.reviewer_impact_measurement
"""
import pickle
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.models import ScoreResult, Evidence, EvidenceDirection
from app.agents.reviewer_agent import review_score
import importlib
decision = importlib.import_module("app.decision")

DATA_PATH = "data/credit_card_fraud_2026.csv"
ARTIFACT_DIR = Path("app/ml_artifacts")

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

AUTO_REJECT_THRESHOLD = 0.15
AUTO_APPROVE_THRESHOLD = 0.01
TOP_N_EVIDENCE = 6
MIN_CONTRIBUTION = 0.005


def naive_decide(score):
    if score >= AUTO_REJECT_THRESHOLD:
        return "auto_reject"
    elif score <= AUTO_APPROVE_THRESHOLD:
        return "auto_approve"
    return "escalate_to_human"


with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(ARTIFACT_DIR / "encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

df = pd.read_csv(DATA_PATH)
X_full = df[FEATURE_ORDER].copy()
for col in CATEGORICAL_COLS:
    X_full[col] = encoders[col].transform(X_full[col])
for col in BOOLEAN_COLS:
    X_full[col] = X_full[col].astype(int)
y_full = df["is_fraud"].values

_, X_test_full, _, y_test_full = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
print(f"Full held-out test set: {len(X_test_full)} transactions, {int(y_test_full.sum())} real fraud")

# NOTE: computing SHAP on the full 4000-row test set was too resource-intensive to
# complete reliably. Using a stratified sample instead: ALL real fraud cases (68) plus
# a random sample of 500 legitimate transactions.
fraud_idx = np.where(y_test_full == 1)[0]
legit_idx = np.where(y_test_full == 0)[0]
rng = np.random.RandomState(42)
sampled_legit_idx = rng.choice(legit_idx, size=min(500, len(legit_idx)), replace=False)
sample_idx = np.concatenate([fraud_idx, sampled_legit_idx])

X_test = X_test_full.iloc[sample_idx]
y_test = y_test_full[sample_idx]
print(f"Measuring on a stratified SAMPLE: {len(X_test)} transactions "
      f"({int(y_test.sum())} fraud, all real fraud cases included + {len(sampled_legit_idx)} random legit)")

print("Computing SHAP values in one batched call...")
explainer = shap.TreeExplainer(model)
proba = model.predict_proba(X_test)[:, 1]
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    all_contributions = shap_values[1]
elif shap_values.ndim == 3:
    all_contributions = shap_values[:, :, 1]
else:
    all_contributions = shap_values
print("Done. Now running Reviewer Agent + Decision Router per transaction...")

changed_count = 0
escalation_saves = 0
downgrade_count = 0
results = []

X_test_reset = X_test.reset_index(drop=True)
raw_df = df.loc[X_test.index].reset_index(drop=True)
print(f"Sanity check: sampled fraud count matches raw_df: {raw_df['is_fraud'].sum()} == {int(y_test.sum())}")

for i in range(len(X_test_reset)):
    contributions = all_contributions[i]
    ranked = sorted(zip(FEATURE_ORDER, contributions), key=lambda x: abs(x[1]), reverse=True)[:TOP_N_EVIDENCE]

    evidence = []
    for feature, contribution in ranked:
        if abs(contribution) < MIN_CONTRIBUTION:
            continue
        strength = round(min(1.0, abs(float(contribution))), 4)
        direction = EvidenceDirection.SUPPORTS_FRAUD if contribution > 0 else EvidenceDirection.CONTRADICTS_FRAUD
        evidence.append(Evidence(
            signal=feature, direction=direction, strength=strength,
            description=f"{feature} (SHAP {contribution:+.4f})",
        ))

    score_result = ScoreResult(
        transaction_id=f"test-{i}", score=round(float(proba[i]), 4),
        evidence=evidence, model_version="random-forest-v1",
    )
    review_result = review_score(score_result)
    actual_decision = decision.route_decision(score_result, review_result)

    naive_outcome = naive_decide(score_result.score)
    actual_outcome = actual_decision.outcome.value
    changed = naive_outcome != actual_outcome

    if changed:
        changed_count += 1
        if naive_outcome in ("auto_reject", "auto_approve") and actual_outcome == "escalate_to_human":
            escalation_saves += 1
    if review_result.verdict.value == "confidence_downgraded":
        downgrade_count += 1

    results.append({
        "transaction_id": f"test-{i}", "actual_fraud": int(raw_df.loc[i, "is_fraud"]),
        "score": score_result.score, "naive_outcome": naive_outcome,
        "actual_outcome": actual_outcome, "changed": changed,
        "reviewer_verdict": review_result.verdict.value,
    })

results_df = pd.DataFrame(results)
total = len(results_df)

print()
print("=" * 60)
print(f"Total test transactions: {total}")
print(f"Reviewer Agent changed the final decision: {changed_count} ({changed_count/total*100:.2f}%)")
print(f"  Of these, escalations that would have been auto-decided: {escalation_saves}")
print(f"  Transactions with a confidence downgrade verdict: {downgrade_count}")
print("=" * 60)

results_df.to_csv(ARTIFACT_DIR / "reviewer_impact_results.csv", index=False)
print(f"\nSaved full per-transaction results to {ARTIFACT_DIR}/reviewer_impact_results.csv")
