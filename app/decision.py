"""
Decision Router.

Combines Agent 2's score with Agent 3's review to produce a final
outcome. This is also where "Check 4: threshold margin" from the
Reviewer Agent spec actually lives -- borderline scores get pushed to
human review here, rather than in Agent 3, because it's a routing
decision, not a confidence judgment.

Thresholds are placeholders for Phase 1, tuned loosely against the
1.7% fraud base rate. Phase 2 replaces these with thresholds justified
by the false-positive cost analysis (a non-negotiable, not polish).
"""

from app.models import ScoreResult, ReviewResult, ReviewVerdict, Decision, DecisionOutcome

AUTO_REJECT_THRESHOLD = 0.75
AUTO_APPROVE_THRESHOLD = 0.15
MARGIN = 0.10  # distance from either threshold that forces escalation


def route_decision(score_result: ScoreResult, review_result: ReviewResult) -> Decision:
    final_score = score_result.score + review_result.confidence_adjustment
    final_score = max(0.0, min(1.0, final_score))

    # A downgraded or insufficient-evidence verdict can never auto-reject,
    # regardless of the raw score -- the whole point of Agent 3 is that a
    # high score with weak evidence should not be actioned automatically.
    if review_result.verdict in (ReviewVerdict.CONFIDENCE_DOWNGRADED, ReviewVerdict.INSUFFICIENT_EVIDENCE):
        if final_score >= AUTO_APPROVE_THRESHOLD:
            return Decision(
                transaction_id=score_result.transaction_id,
                outcome=DecisionOutcome.ESCALATE_TO_HUMAN,
                final_score=round(final_score, 4),
                reason=f"Reviewer verdict '{review_result.verdict.value}' blocks auto-reject; "
                       f"routed to human review. Reviewer reason: {review_result.reason}",
            )

    # Check 4: threshold margin. Anything within MARGIN of either cutoff
    # is treated as borderline and escalated rather than auto-decided.
    near_reject_line = abs(final_score - AUTO_REJECT_THRESHOLD) <= MARGIN
    near_approve_line = abs(final_score - AUTO_APPROVE_THRESHOLD) <= MARGIN
    if near_reject_line or near_approve_line:
        return Decision(
            transaction_id=score_result.transaction_id,
            outcome=DecisionOutcome.ESCALATE_TO_HUMAN,
            final_score=round(final_score, 4),
            reason=f"Final score {final_score:.2f} sits within the borderline margin "
                   f"({MARGIN}) of a decision threshold -- routed to human review.",
        )

    if final_score >= AUTO_REJECT_THRESHOLD:
        outcome = DecisionOutcome.AUTO_REJECT
        reason = f"Final score {final_score:.2f} at/above reject threshold {AUTO_REJECT_THRESHOLD}, evidence upheld."
    elif final_score <= AUTO_APPROVE_THRESHOLD:
        outcome = DecisionOutcome.AUTO_APPROVE
        reason = f"Final score {final_score:.2f} at/below approve threshold {AUTO_APPROVE_THRESHOLD}."
    else:
        outcome = DecisionOutcome.ESCALATE_TO_HUMAN
        reason = f"Final score {final_score:.2f} between thresholds -- routed to human review."

    return Decision(
        transaction_id=score_result.transaction_id,
        outcome=outcome,
        final_score=round(final_score, 4),
        reason=reason,
    )
