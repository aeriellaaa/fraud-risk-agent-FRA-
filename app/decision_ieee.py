"""
Decision Router (IEEE version). Thresholds are NOT pure cost-optimal -- an earlier sweep
found the unconstrained cost-optimal threshold flags 71% of all transactions, which is
operationally undeployable. Instead, AUTO_REJECT_THRESHOLD is set at the 2% false-positive
budget operating point (see scripts/threshold_budget_analysis.py output), matching kavach's
own "recall at a fixed FP budget" methodology rather than chasing a raw cost minimum.
"""
from app.models import ScoreResult, ReviewResult, ReviewVerdict, Decision, DecisionOutcome

AUTO_REJECT_THRESHOLD = 0.7574  # 2% FP-budget operating point, see threshold_budget_analysis.py
AUTO_APPROVE_THRESHOLD = 0.12   # near the legit-transaction median score
MARGIN = 0.02


def route_decision_ieee(score_result: ScoreResult, review_result: ReviewResult) -> Decision:
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
