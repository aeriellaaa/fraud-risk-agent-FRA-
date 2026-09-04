"""
Agent 3 -- Reviewer Agent (Evidence-Strength Check), recalibrated for the
ML scorer. Evidence strength is a raw SHAP contribution magnitude
(typically 0.005-0.15), NOT a 0-1 relative score.
"""

from app.models import ScoreResult, ReviewResult, ReviewVerdict, EvidenceDirection

SIGNAL_CATEGORY = {
    "amount_usd": "financial", "account_balance_usd": "financial",
    "merchant_category": "merchant", "merchant_risk_score": "merchant", "is_new_merchant": "merchant",
    "card_type": "card", "auth_method": "card", "card_age_months": "card",
    "channel": "device", "device_type": "device", "used_vpn": "device",
    "is_foreign_transaction": "location", "distance_from_home_km": "location",
    "ip_country_mismatch": "location", "billing_shipping_mismatch": "location",
    "hours_since_last_txn": "behavior", "txn_count_last_24h": "behavior",
    "velocity_score": "behavior", "time_of_day_hour": "behavior", "day_of_week": "behavior",
    "cvv_retry_count": "authentication",
    "customer_age": "profile",
    "is_ai_generated_scam_attempt": "ai_flag",
    "prior_disputes": "dispute_history",
    "pattern_evasion": "pattern_evasion",
}

HIGH_SCORE_THRESHOLD = 0.01   # real cost-optimal threshold, sourced Rs94 FP / Rs34802 FN (see scripts/train_model.py)
DOWNGRADE_THRESHOLD = -0.05


def review_score(score_result: ScoreResult) -> ReviewResult:
    supporting = [e for e in score_result.evidence if e.direction == EvidenceDirection.SUPPORTS_FRAUD]
    contradicting = [e for e in score_result.evidence if e.direction == EvidenceDirection.CONTRADICTS_FRAUD]

    adjustment = 0.0
    reasons = []

    if score_result.score >= HIGH_SCORE_THRESHOLD and len(supporting) <= 1:
        adjustment -= 0.05
        reasons.append(
            f"Score of {score_result.score:.4f} rests on only {len(supporting)} "
            f"supporting signal(s) -- flagged score, thin evidence."
        )

    categories = {SIGNAL_CATEGORY.get(e.signal, e.signal) for e in supporting}
    diversity_ratio = (len(categories) / len(supporting)) if supporting else 1.0
    if len(supporting) >= 2 and diversity_ratio < 0.5:
        adjustment -= 0.02
        reasons.append(
            f"Supporting evidence concentrated in {len(categories)} category(ies) "
            f"across {len(supporting)} signals -- limited independent corroboration."
        )

    # Contradiction penalty is RELATIVE to supporting strength, not absolute --
    # a real fraud case can have both supporting and contradicting signals; what
    # matters is whether contradiction outweighs support, not whether it merely exists.
    supporting_strength = sum(e.strength for e in supporting)
    contradiction_strength = sum(e.strength for e in contradicting)
    if contradiction_strength > 0:
        net_contradiction = contradiction_strength - supporting_strength
        if net_contradiction > 0:
            penalty = min(0.05, net_contradiction * 0.3)
            adjustment -= penalty
            reasons.append(
                f"{len(contradicting)} contradicting signal(s) (magnitude {contradiction_strength:.4f}) "
                f"outweigh supporting evidence (magnitude {supporting_strength:.4f})."
            )

    adjustment = max(-1.0, min(0.0, adjustment))

    if len(supporting) == 0:
        verdict = ReviewVerdict.INSUFFICIENT_EVIDENCE
        prefix = "No supporting evidence found for a fraud determination."
    elif adjustment <= DOWNGRADE_THRESHOLD:
        verdict = ReviewVerdict.CONFIDENCE_DOWNGRADED
        prefix = "Confidence downgraded:"
    else:
        verdict = ReviewVerdict.CONFIDENCE_UPHELD
        prefix = "Confidence upheld."
        if not reasons:
            reasons.append("Evidence count, diversity, and consistency all support the score as computed.")

    full_reason = prefix + (" " + "; ".join(reasons) if reasons else "")

    return ReviewResult(
        transaction_id=score_result.transaction_id,
        verdict=verdict,
        confidence_adjustment=round(adjustment, 4),
        reason=full_reason,
    )
