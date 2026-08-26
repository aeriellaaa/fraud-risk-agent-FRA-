"""
Agent 3 -- Reviewer Agent (Evidence-Strength Check).

Takes Agent 2's score + evidence list and checks whether the score's
confidence is actually earned -- NOT a second opinion on fraud, a check
on whether the first opinion is trustworthy. Implements the four locked
checks from the roadmap:

  1. Evidence count vs score magnitude
  2. Evidence diversity (redundant same-type signals count for less)
  3. Contradiction check
  4. Threshold margin (handled downstream by the Decision Router, which
     reads this agent's adjusted score to decide auto vs escalate)
"""

from app.models import ScoreResult, ReviewResult, ReviewVerdict, EvidenceDirection

# Groups signals into independent categories. Two signals from the same
# category (e.g. cvv_risk and device_risk, both "device") corroborate
# each other less than two signals from different categories.
SIGNAL_CATEGORY = {
    "geo_mismatch": "location",
    "cvv_risk": "device",
    "device_risk": "device",
    "velocity_flag": "behavior",
    "high_amount_ratio": "behavior",
    "merchant_risk_high": "merchant",
    "new_merchant_risk": "merchant",
    "prior_dispute_risk": "dispute_history",
    "ai_scam_flag": "ai_flag",
}

HIGH_SCORE_THRESHOLD = 0.5
DOWNGRADE_THRESHOLD = -0.25


def review_score(score_result: ScoreResult) -> ReviewResult:
    supporting = [e for e in score_result.evidence if e.direction == EvidenceDirection.SUPPORTS_FRAUD]
    contradicting = [e for e in score_result.evidence if e.direction == EvidenceDirection.CONTRADICTS_FRAUD]

    adjustment = 0.0
    reasons = []

    # Check 1: evidence count vs score magnitude
    if score_result.score >= HIGH_SCORE_THRESHOLD and len(supporting) <= 1:
        adjustment -= 0.35
        reasons.append(
            f"Score of {score_result.score:.2f} rests on only {len(supporting)} "
            f"supporting signal(s) -- high score, thin evidence."
        )

    # Check 2: evidence diversity
    categories = {SIGNAL_CATEGORY.get(e.signal, e.signal) for e in supporting}
    diversity_ratio = (len(categories) / len(supporting)) if supporting else 1.0
    if len(supporting) >= 2 and diversity_ratio < 0.6:
        adjustment -= 0.15
        reasons.append(
            f"Supporting evidence concentrated in {len(categories)} category(ies) "
            f"across {len(supporting)} signals -- limited independent corroboration."
        )

    # Check 3: contradiction check
    contradiction_strength = sum(e.strength for e in contradicting)
    if contradiction_strength > 0:
        penalty = min(0.30, contradiction_strength * 0.5)
        adjustment -= penalty
        reasons.append(
            f"{len(contradicting)} contradicting signal(s) found "
            f"(total strength {contradiction_strength:.2f}) -- confidence downgraded."
        )

    adjustment = max(-1.0, min(0.0, adjustment))

    # Verdict
    if len(supporting) == 0:
        verdict = ReviewVerdict.INSUFFICIENT_EVIDENCE
        if not reasons:
            reasons.append("No supporting evidence found for a fraud determination.")
    elif adjustment <= DOWNGRADE_THRESHOLD:
        verdict = ReviewVerdict.CONFIDENCE_DOWNGRADED
    else:
        verdict = ReviewVerdict.CONFIDENCE_UPHELD
        if not reasons:
            reasons.append("Evidence count, diversity, and consistency all support the score as computed.")

    return ReviewResult(
        transaction_id=score_result.transaction_id,
        verdict=verdict,
        confidence_adjustment=round(adjustment, 4),
        reason="; ".join(reasons),
    )
