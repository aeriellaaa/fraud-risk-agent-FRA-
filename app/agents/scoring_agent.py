"""
Agent 2 -- Detection & Scoring Agent.

Rule-based scorer (Phase 1). Swap for LightGBM in Phase 2 -- the output
shape (ScoreResult with an evidence list) stays the same so Agent 3
does not need to change when the scorer changes underneath it.

Evidence weights below reflect how discriminative each signal actually
is in credit_card_fraud_2026.csv (see features.py for the underlying
fraud-rate/mean comparisons), not arbitrary guesses. Roughly:
  strong  (0.6-0.7): geo_mismatch, cvv_risk, ai_scam_flag
  medium  (0.4-0.5): velocity_flag, device_risk, merchant_risk high
  weak    (0.2-0.3): new_merchant_risk, prior_dispute_risk, high_amount_ratio
"""

from app.models import FeatureVector, ScoreResult, Evidence, EvidenceDirection

WEIGHTS = {
    "geo_mismatch": 0.70,
    "cvv_risk": 0.60,
    "ai_scam_flag": 0.75,
    "velocity_flag": 0.45,
    "device_risk": 0.45,
    "merchant_risk_high": 0.45,
    "new_merchant_risk": 0.30,
    "prior_dispute_risk": 0.30,
    "high_amount_ratio": 0.20,
}

MERCHANT_RISK_HIGH = 47.3
MERCHANT_RISK_LOW = 15.0  # used only to generate contradicting evidence


def score_transaction(features: FeatureVector) -> ScoreResult:
    evidence: list[Evidence] = []
    triggered_weight = 0.0
    total_weight = sum(WEIGHTS.values())

    def add_supporting(key: str, description: str) -> None:
        nonlocal triggered_weight
        w = WEIGHTS[key]
        triggered_weight += w
        evidence.append(Evidence(
            signal=key,
            direction=EvidenceDirection.SUPPORTS_FRAUD,
            strength=w,
            description=description,
        ))

    if features.geo_mismatch:
        add_supporting("geo_mismatch", "IP/billing/foreign location mismatch detected")
    if features.cvv_risk:
        add_supporting("cvv_risk", "One or more CVV retries on this transaction")
    if features.ai_scam_flag:
        add_supporting("ai_scam_flag", "Transaction flagged as a likely AI-generated scam attempt")
    if features.velocity_flag:
        add_supporting("velocity_flag", "Transaction velocity above typical range")
    if features.device_risk:
        add_supporting("device_risk", "VPN usage detected on this transaction")
    if features.merchant_risk_score > MERCHANT_RISK_HIGH:
        add_supporting("merchant_risk_high", f"Merchant risk score {features.merchant_risk_score:.1f} is above threshold")
    if features.new_merchant_risk:
        add_supporting("new_merchant_risk", "First transaction with this merchant")
    if features.prior_dispute_risk:
        add_supporting("prior_dispute_risk", "Cardholder has prior disputes on record")
    if features.high_amount_ratio:
        add_supporting("high_amount_ratio", "Transaction amount is large relative to account balance")

    # Contradicting evidence -- gives Agent 3's contradiction check real
    # material to work with, instead of only ever seeing one-directional
    # evidence.
    if features.merchant_risk_score < MERCHANT_RISK_LOW:
        evidence.append(Evidence(
            signal="merchant_risk_low",
            direction=EvidenceDirection.CONTRADICTS_FRAUD,
            strength=0.30,
            description=f"Merchant risk score {features.merchant_risk_score:.1f} is well below typical risk range",
        ))
    if not features.geo_mismatch and not features.device_risk and not features.cvv_risk:
        evidence.append(Evidence(
            signal="clean_device_and_location",
            direction=EvidenceDirection.CONTRADICTS_FRAUD,
            strength=0.25,
            description="No location, device, or CVV anomalies detected",
        ))

    score = min(1.0, triggered_weight / total_weight)

    return ScoreResult(
        transaction_id=features.transaction_id,
        score=round(score, 4),
        evidence=evidence,
        model_version="rule-based-v0",
    )
