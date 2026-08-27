from app.decision import route_decision
from app.models import (
    ScoreResult,
    ReviewResult,
    ReviewVerdict,
    DecisionOutcome,
)

def test_low_score_is_auto_approved():
    score_result = ScoreResult(
        transaction_id="decision-test-001",
        score=0.00,
        evidence=[],
    )

    review_result = ReviewResult(
        transaction_id="decision-test-001",
        verdict=ReviewVerdict.CONFIDENCE_UPHELD,
        confidence_adjustment=0.0,
        reason="Evidence supports the score.",
    )

    result = route_decision(score_result, review_result)

    assert result.outcome == DecisionOutcome.AUTO_APPROVE
    assert result.final_score == 0.00
def test_high_trusted_score_is_auto_rejected():
    score_result = ScoreResult(
        transaction_id="decision-test-002",
        score=0.90,
        evidence=[],
    )

    review_result = ReviewResult(
        transaction_id="decision-test-002",
        verdict=ReviewVerdict.CONFIDENCE_UPHELD,
        confidence_adjustment=0.0,
        reason="Strong evidence supports the score.",
    )

    result = route_decision(score_result, review_result)

    assert result.outcome == DecisionOutcome.AUTO_REJECT
    assert result.final_score == 0.90
def test_borderline_score_is_escalated():
    score_result = ScoreResult(
        transaction_id="decision-test-003",
        score=0.20,
        evidence=[],
    )

    review_result = ReviewResult(
        transaction_id="decision-test-003",
        verdict=ReviewVerdict.CONFIDENCE_UPHELD,
        confidence_adjustment=0.0,
        reason="Evidence supports the score.",
    )

    result = route_decision(score_result, review_result)

    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN
def test_downgraded_high_score_cannot_auto_reject():
    score_result = ScoreResult(
        transaction_id="decision-test-004",
        score=0.90,
        evidence=[],
    )

    review_result = ReviewResult(
        transaction_id="decision-test-004",
        verdict=ReviewVerdict.CONFIDENCE_DOWNGRADED,
        confidence_adjustment=-0.05,
        reason="High score but insufficiently diverse evidence.",
    )

    result = route_decision(score_result, review_result)

    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN
    assert result.final_score == 0.85