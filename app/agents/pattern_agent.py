"""
Agent 1 -- Pattern/Evasion Agent.

IMPORTANT (defensive framing, non-negotiable): this agent performs
DEFENSIVE DRIFT AND EVASION DETECTION ONLY. It does not model, simulate,
or generate fraud techniques. It flags statistical irregularities and
threshold-skirting behavior in incoming transactions -- nothing here
constructs or teaches fraud strategies.

Redesigned from the original per-card drift design: the dataset has no
card_id linking transactions to the same cardholder over time, so
"compare this card to its own history" is not possible. Replaced with
two techniques that do not require history:

  1. Population-level multivariate anomaly detection -- flags
     transactions where multiple discriminative signals are
     simultaneously elevated relative to the dataset baseline, since
     co-occurring anomalies are rarer and more suspicious than any
     single elevated signal alone.

  2. Threshold-evasion detection -- flags transactions sitting just
     under Agent 2's known scoring thresholds (velocity, merchant
     risk). A transaction engineered to stay just below a known cutoff
     is a classic adversarial evasion tell, distinct from a fraud
     score itself.

Baseline stats (mean, std) are computed from credit_card_fraud_2026.csv
directly, not guessed:
  velocity_score:       mean=19.81, std=12.37
  txn_count_last_24h:   mean=3.19,  std=1.78
  merchant_risk_score:  mean=37.40, std=17.06
  cvv_retry_count:      mean=0.18,  std=0.42
"""

from app.models import TransactionIn, DriftResult

BASELINE = {
    "velocity_score": (19.81, 12.37),
    "txn_count_last_24h": (3.19, 1.78),
    "merchant_risk_score": (37.40, 17.06),
    "cvv_retry_count": (0.18, 0.42),
}

Z_SCORE_ELEVATED = 1.5
MULTIVARIATE_MIN_SIGNALS = 2

# Same thresholds Agent 2 uses -- imported as values, not by importing
# scoring_agent, to keep Agent 1 independent of Agent 2's implementation.
VELOCITY_THRESHOLD = 27.6
MERCHANT_RISK_THRESHOLD = 47.3
EVASION_MARGIN_VELOCITY = 3.0
EVASION_MARGIN_MERCHANT_RISK = 5.0


def _zscore(value: float, key: str) -> float:
    mean, std = BASELINE[key]
    return (value - mean) / std if std > 0 else 0.0


def detect_drift(txn: TransactionIn) -> DriftResult:
    signals: list[str] = []

    # --- 1. Population-level multivariate anomaly ---
    elevated = []
    for key in ("velocity_score", "txn_count_last_24h", "merchant_risk_score", "cvv_retry_count"):
        value = getattr(txn, key)
        z = _zscore(value, key)
        if z > Z_SCORE_ELEVATED:
            elevated.append(key)

    if len(elevated) >= MULTIVARIATE_MIN_SIGNALS:
        signals.append(
            f"multivariate_anomaly: {len(elevated)} signals simultaneously elevated "
            f"({', '.join(elevated)})"
        )

    # --- 2. Threshold-evasion detection ---
    if VELOCITY_THRESHOLD - EVASION_MARGIN_VELOCITY <= txn.velocity_score < VELOCITY_THRESHOLD:
        signals.append(
            f"velocity_evasion: velocity_score {txn.velocity_score:.1f} sits just under "
            f"the {VELOCITY_THRESHOLD} scoring threshold"
        )

    if MERCHANT_RISK_THRESHOLD - EVASION_MARGIN_MERCHANT_RISK <= txn.merchant_risk_score < MERCHANT_RISK_THRESHOLD:
        signals.append(
            f"merchant_risk_evasion: merchant_risk_score {txn.merchant_risk_score:.1f} sits "
            f"just under the {MERCHANT_RISK_THRESHOLD} scoring threshold"
        )

    # Drift score: proportion of checks triggered, capped at 1.0.
    # 2 possible signal families (multivariate, evasion), each contributes
    # up to 0.5, scaled by how many concrete signals fired within it.
    drift_score = 0.0
    if elevated:
        drift_score += 0.5 * min(1.0, len(elevated) / 3)
    evasion_hits = sum(1 for s in signals if "evasion" in s)
    if evasion_hits:
        drift_score += 0.5 * min(1.0, evasion_hits / 2)

    return DriftResult(
        transaction_id=txn.transaction_id,
        drift_score=round(min(1.0, drift_score), 4),
        drift_signals=signals,
    )
