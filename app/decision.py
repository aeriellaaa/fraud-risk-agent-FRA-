"""
Decision Router, recalibrated for the ML scorer's real probability range
(max observed on held-out test: 0.28) and the cost-optimal threshold
found in scripts/train_model.py (0.02, using placeholder $25 FP / $2000
FN costs -- not yet verified citations, see README).
"""

from app.models import ScoreResult, ReviewResult, ReviewVerdict, Decision, DecisionOutcome

AUTO_REJECT_THRESHOLD = 0.15   # conservative: near the top of the observed score range
AUTO_APPROVE_THRESHOLD = 0.01  # real cost-optimal threshold, sourced Rs94 FP / Rs34802 FN (see scripts/train_model.py)
MARGIN = 0.01


def route_decision(score_result: ScoreResult, review_result: ReviewResult) -> Decision:
    final_score = score_result.score + review_result.confidence_adjustment
    final_score = max(0.0, min(1.0, final_score))

    if review_result.verdict in (ReviewVerdict.CONFIDENCE_DOWNGRADED, ReviewVerdict.INSUFFICIENT_EVIDENCE):
        if final_score >= AUTO_APPROVE_THRESHOLD:
            return Decision(
                transaction_id=score_result.transaction_id,
                outcome=DecisionOutcome.ESCALATE_TO_HUMAN,
                final_score=round(final_score, 4),
                reason=f"Reviewer verdict '{review_result.verdict.value}' blocks auto-reject; "
                       f"routed to human review. Reviewer reason: {review_result.reason}",
            )

    near_reject_line = abs(final_score - AUTO_REJECT_THRESHOLD) <= MARGIN
    near_approve_line = abs(final_score - AUTO_APPROVE_THRESHOLD) <= MARGIN
    if near_reject_line or near_approve_line:
        return Decision(
            transaction_id=score_result.transaction_id,
            outcome=DecisionOutcome.ESCALATE_TO_HUMAN,
            final_score=round(final_score, 4),
            reason=f"Final score {final_score:.4f} sits within the borderline margin "
                   f"({MARGIN}) of a decision threshold -- routed to human review.",
        )

    if final_score >= AUTO_REJECT_THRESHOLD:
        outcome = DecisionOutcome.AUTO_REJECT
        reason = f"Final score {final_score:.4f} at/above reject threshold {AUTO_REJECT_THRESHOLD}, evidence upheld."
    elif final_score <= AUTO_APPROVE_THRESHOLD:
        outcome = DecisionOutcome.AUTO_APPROVE
        reason = f"Final score {final_score:.4f} at/below approve threshold {AUTO_APPROVE_THRESHOLD}."
    else:
        outcome = DecisionOutcome.ESCALATE_TO_HUMAN
        reason = f"Final score {final_score:.4f} between thresholds -- routed to human review."

    return Decision(
        transaction_id=score_result.transaction_id,
        outcome=outcome,
        final_score=round(final_score, 4),
        reason=reason,
    )
