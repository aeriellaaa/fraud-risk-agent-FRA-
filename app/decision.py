"""
Decision Router, recalibrated for the ML scorer's real probability range
(max observed on held-out test: 0.28) and the cost-optimal threshold
found in scripts/train_model.py (0.01, using verified Rs94 FP / Rs34802 FN
costs -- FN cost sourced to Lok Sabha data via Business Standard, FP cost
estimated from ERI SalaryExpert; see docs/model_selection.md for full
sourcing detail and which figure is verified vs. estimated).

CAPACITY-AWARE MODE: the 0.01 cost-optimal threshold assumes unlimited review
capacity. Verified against the real model: at 5,000 txns/day, it escalates 30%
of traffic (95.6% recall) -- no team up to 20 analysts can sustain that rate.
See app/capacity.py and scripts/capacity_cross_check.py for the full finding.

Set the TEAM_SIZE environment variable to switch AUTO_APPROVE_THRESHOLD from the
cost-optimal default to a capacity-respecting value computed from app.capacity,
using this project's own sourced review-time assumption (~14 min/review). This
is a staffing decision, not a silent default -- unset TEAM_SIZE to keep the
original cost-optimal behavior.
"""

import os
from app.models import ScoreResult, ReviewResult, ReviewVerdict, Decision, DecisionOutcome

COST_OPTIMAL_AUTO_APPROVE_THRESHOLD = 0.01  # sourced Rs94 FP / Rs34802 FN (see scripts/train_model.py)
AUTO_REJECT_THRESHOLD = 0.15   # conservative: near the top of the observed score range

_team_size_env = os.getenv("TEAM_SIZE")
if _team_size_env:
    import json
    from pathlib import Path

    _CAPACITY_CACHE_PATH = Path("app/ml_artifacts/capacity_cross_check_results.json")
    if not _CAPACITY_CACHE_PATH.exists():
        raise FileNotFoundError(
            "TEAM_SIZE is set but app/ml_artifacts/capacity_cross_check_results.json doesn't "
            "exist yet. Run: python -m scripts.capacity_cross_check   -- or unset TEAM_SIZE."
        )
    with open(_CAPACITY_CACHE_PATH) as f:
        _cached = json.load(f)
    _row = next((r for r in _cached["rows"] if r.get("team_size") == int(_team_size_env)), None)
    if not (_row and _row["sustainable"]):
        raise KeyError(
            f"TEAM_SIZE={_team_size_env} not found in cached results or not sustainable at the "
            f"assumed transaction volume -- re-run scripts/capacity_cross_check.py with this "
            f"team size, or unset TEAM_SIZE to use the cost-optimal default."
        )
    AUTO_APPROVE_THRESHOLD = _row["auto_approve_threshold"]
    _THRESHOLD_SOURCE = (
        f"capacity-constrained (TEAM_SIZE={_team_size_env}, escalate_rate={_row['escalate_rate']:.2%}, "
        f"recall={_row['recall']:.1%})"
    )
else:
    AUTO_APPROVE_THRESHOLD = COST_OPTIMAL_AUTO_APPROVE_THRESHOLD
    _THRESHOLD_SOURCE = "cost-optimal default (TEAM_SIZE unset -- assumes unlimited review capacity)"

MARGIN = 0.002  # tightened for the real ML score range (0-0.28); 0.01 was inherited from the placeholder-threshold era and was swallowing legitimate low scores into escalation


def route_decision(score_result: ScoreResult, review_result: ReviewResult) -> Decision:
    final_score = score_result.score + review_result.confidence_adjustment
    final_score = max(0.0, min(1.0, final_score))

    # SAFETY NET: if the Reviewer Agent downgraded confidence or found insufficient
    # evidence, and the ORIGINAL (pre-adjustment) score wasn't already clearly safe,
    # force human review -- regardless of which direction the adjustment happened to
    # push the final score. Checked against score_result.score, NOT final_score.
    #
    # BUG FOUND (see scripts/reviewer_impact_measurement.py): the previous version only
    # checked final_score >= AUTO_APPROVE_THRESHOLD, which blocked a downgraded score
    # from being auto-REJECTED, but did nothing if the SAME downgrade pushed the score
    # below the approve threshold instead. On a 568-transaction sample, this let 39 of
    # 68 real fraud cases (57%) get auto-approved outright.
    if review_result.verdict in (ReviewVerdict.CONFIDENCE_DOWNGRADED, ReviewVerdict.INSUFFICIENT_EVIDENCE):
        if score_result.score >= AUTO_APPROVE_THRESHOLD:
            return Decision(
                transaction_id=score_result.transaction_id,
                outcome=DecisionOutcome.ESCALATE_TO_HUMAN,
                final_score=round(final_score, 4),
                reason=f"Reviewer verdict '{review_result.verdict.value}' blocks both auto-reject "
                       f"and auto-approve; original score {score_result.score:.4f} was not already "
                       f"safe on its own -- routed to human review. Reviewer reason: {review_result.reason}",
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
    elif score_result.score <= AUTO_APPROVE_THRESHOLD:
        # Auto-approve requires the ORIGINAL score to already be safe on its own --
        # a negative reviewer adjustment must never be what PUSHES a transaction into
        # approval from a higher original score.
        outcome = DecisionOutcome.AUTO_APPROVE
        reason = f"Original score {score_result.score:.4f} was already at/below approve " \
                 f"threshold {AUTO_APPROVE_THRESHOLD} before any reviewer adjustment."
    else:
        outcome = DecisionOutcome.ESCALATE_TO_HUMAN
        reason = f"Final score {final_score:.4f} between thresholds -- routed to human review."

    return Decision(
        transaction_id=score_result.transaction_id,
        outcome=outcome,
        final_score=round(final_score, 4),
        reason=reason,
    )
