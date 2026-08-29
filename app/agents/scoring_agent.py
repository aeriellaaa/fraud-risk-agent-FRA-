"""
Agent 2 -- ML-based scoring, using the trained Random Forest + SHAP.

Evidence strength is the RAW absolute SHAP contribution (not normalized
to the top signal within a transaction). This keeps strength directly
comparable to the model's score itself and to other transactions --
normalizing per-transaction was inflating minor SHAP noise into
misleadingly large "contradiction" signals.
"""

import pickle
import shap
import numpy as np
from pathlib import Path

from app.models import TransactionIn, ScoreResult, Evidence, EvidenceDirection, DriftResult

ARTIFACT_DIR = Path("app/ml_artifacts")

with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
    _model = pickle.load(f)
with open(ARTIFACT_DIR / "encoders.pkl", "rb") as f:
    _encoders = pickle.load(f)

_explainer = shap.TreeExplainer(_model)

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

FEATURE_DESCRIPTIONS = {
    "amount_usd": "Transaction amount", "merchant_category": "Merchant category",
    "card_type": "Card type", "auth_method": "Authentication method",
    "channel": "Transaction channel", "device_type": "Device type",
    "is_foreign_transaction": "Foreign transaction flag",
    "hours_since_last_txn": "Time since last transaction",
    "txn_count_last_24h": "Transaction count in last 24h",
    "distance_from_home_km": "Distance from home", "card_age_months": "Card age",
    "customer_age": "Customer age", "account_balance_usd": "Account balance",
    "is_new_merchant": "First transaction with this merchant", "used_vpn": "VPN usage",
    "ip_country_mismatch": "IP/country mismatch",
    "billing_shipping_mismatch": "Billing/shipping mismatch",
    "cvv_retry_count": "CVV retry count", "velocity_score": "Velocity score",
    "time_of_day_hour": "Time of day", "day_of_week": "Day of week",
    "is_ai_generated_scam_attempt": "AI-generated scam attempt flag",
    "merchant_risk_score": "Merchant risk score", "prior_disputes": "Prior disputes on record",
}

TOP_N_EVIDENCE = 6
MIN_CONTRIBUTION = 0.005


def _encode_transaction(txn: TransactionIn) -> np.ndarray:
    row = []
    for col in FEATURE_ORDER:
        val = getattr(txn, col)
        if col in CATEGORICAL_COLS:
            le = _encoders[col]
            val = le.transform([val])[0] if val in le.classes_ else -1
        elif col in BOOLEAN_COLS:
            val = int(val)
        row.append(val)
    return np.array(row).reshape(1, -1)


def score_transaction(txn: TransactionIn, drift: DriftResult | None = None) -> ScoreResult:
    X = _encode_transaction(txn)
    proba = float(_model.predict_proba(X)[0, 1])

    shap_values = _explainer.shap_values(X)
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    elif shap_values.ndim == 3:
        contributions = shap_values[0, :, 1]
    else:
        contributions = shap_values[0]

    ranked = sorted(
        zip(FEATURE_ORDER, contributions),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:TOP_N_EVIDENCE]

    evidence: list[Evidence] = []
    for feature, contribution in ranked:
        if abs(contribution) < MIN_CONTRIBUTION:
            continue
        strength = round(min(1.0, abs(float(contribution))), 4)
        direction = EvidenceDirection.SUPPORTS_FRAUD if contribution > 0 else EvidenceDirection.CONTRADICTS_FRAUD
        value = getattr(txn, feature)
        evidence.append(Evidence(
            signal=feature,
            direction=direction,
            strength=strength,
            description=f"{FEATURE_DESCRIPTIONS.get(feature, feature)}: {value} "
                        f"(SHAP contribution {contribution:+.4f})",
        ))

    if drift is not None and drift.drift_score >= 0.3:
        evidence.append(Evidence(
            signal="pattern_evasion",
            direction=EvidenceDirection.SUPPORTS_FRAUD,
            strength=round(min(1.0, drift.drift_score * 0.1), 4),
            description=f"Agent 1 flagged pattern/evasion signals: {'; '.join(drift.drift_signals)}",
        ))

    return ScoreResult(
        transaction_id=txn.transaction_id,
        score=round(proba, 4),
        evidence=evidence,
        model_version="random-forest-v1",
    )
