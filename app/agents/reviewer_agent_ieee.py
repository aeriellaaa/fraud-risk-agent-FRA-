"""
Agent 3 (IEEE version) -- same evidence-strength check design as the original
reviewer_agent.py, recalibrated for the IEEE model's feature names and its much wider,
better-separated probability range (max ~0.9995, vs the old model's compressed ~0.28).
"""
from app.models import ScoreResult, ReviewResult, ReviewVerdict, EvidenceDirection

CATEGORY_PREFIXES = {
    "card": "card", "TransactionAmt": "financial", "addr": "location", "dist": "location",
    "P_email": "email", "R_email": "email", "C1": "velocity_agg", "C2": "velocity_agg",
    "C3": "velocity_agg", "C4": "velocity_agg", "C5": "velocity_agg", "C6": "velocity_agg",
    "C7": "velocity_agg", "C8": "velocity_agg", "C9": "velocity_agg", "C10": "velocity_agg",
    "C11": "velocity_agg", "C12": "velocity_agg", "C13": "velocity_agg", "C14": "velocity_agg",
    "D1": "time_delta", "D2": "time_delta", "D4": "time_delta", "D10": "time_delta",
    "D15": "time_delta", "M": "match_flags", "hour_of_day": "behavior", "day_of_week": "behavior",
    "DeviceType": "device", "DeviceInfo": "device", "has_identity_data": "device",
    "id_": "identity", "engineered_signals_combined": "engineered", "pattern_evasion": "pattern_evasion",
    "ProductCD": "merchant",
}

def _category(signal: str) -> str:
    for prefix, cat in CATEGORY_PREFIXES.items():
        if signal.startswith(prefix):
            return cat
    return "other"

# Recalibrated for the IEEE model's real probability range (see docs/ieee_cis_migration_log.md)
HIGH_SCORE_THRESHOLD = 0.3
DOWNGRADE_THRESHOLD = -0.08


def review_score_ieee(score_result: ScoreResult) -> ReviewResult:
    supporting = [e for e in score_result.evidence if e.direction == EvidenceDirection.SUPPORTS_FRAUD]
    contradicting = [e for e in score_result.evidence if e.direction == EvidenceDirection.CONTRADICTS_FRAUD]

    adjustment = 0.0
    reasons = []

    if score_result.score >= HIGH_SCORE_THRESHOLD and len(supporting) <= 1:
        adjustment -= 0.08
        reasons.append(
            f"Score of {score_result.score:.4f} rests on only {len(supporting)} "
            f"supporting signal(s) -- flagged score, thin evidence."
        )

    categories = {_category(e.signal) for e in supporting}
    diversity_ratio = (len(categories) / len(supporting)) if supporting else 1.0
    if len(supporting) >= 2 and diversity_ratio < 0.4:
        adjustment -= 0.04
        reasons.append(
            f"Supporting evidence concentrated in {len(categories)} category(ies) "
            f"across {len(supporting)} signals -- limited independent corroboration."
        )

    contradiction_strength = sum(e.strength for e in contradicting)
    if contradiction_strength > 0:
        penalty = min(0.08, contradiction_strength * 0.3)
        adjustment -= penalty
        reasons.append(
            f"{len(contradicting)} contradicting signal(s) found "
            f"(total SHAP magnitude {contradiction_strength:.4f})."
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
