"""
Tests for the Decision Router. Written against real thresholds:
AUTO_APPROVE_THRESHOLD=0.01, AUTO_REJECT_THRESHOLD=0.15, MARGIN=0.002.
"""

from app.decision import route_decision
from app.models import ScoreResult, ReviewResult, ReviewVerdict, DecisionOutcome


def test_low_score_is_auto_approved():
    score_result = ScoreResult(transaction_id="d-001", score=0.0, evidence=[],
                                model_version="random-forest-v1")
    review_result = ReviewResult(transaction_id="d-001", verdict=ReviewVerdict.CONFIDENCE_UPHELD,
                                  confidence_adjustment=0.0, reason="No red flags.")
    result = route_decision(score_result, review_result)
    assert result.outcome == DecisionOutcome.AUTO_APPROVE
    assert result.final_score == 0.0


def test_high_trusted_score_is_auto_rejected():
    score_result = ScoreResult(transaction_id="d-002", score=0.20, evidence=[],
                                model_version="random-forest-v1")
    review_result = ReviewResult(transaction_id="d-002", verdict=ReviewVerdict.CONFIDENCE_UPHELD,
                                  confidence_adjustment=0.0, reason="Strong evidence.")
    result = route_decision(score_result, review_result)
    assert result.outcome == DecisionOutcome.AUTO_REJECT
    assert result.final_score == 0.20


def test_borderline_score_is_escalated():
    score_result = ScoreResult(transaction_id="d-003", score=0.08, evidence=[],
                                model_version="random-forest-v1")
    review_result = ReviewResult(transaction_id="d-003", verdict=ReviewVerdict.CONFIDENCE_UPHELD,
                                  confidence_adjustment=0.0, reason="Evidence supports the score.")
    result = route_decision(score_result, review_result)
    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN


def test_downgraded_high_score_cannot_auto_reject():
    score_result = ScoreResult(transaction_id="d-004", score=0.20, evidence=[],
                                model_version="random-forest-v1")
    review_result = ReviewResult(transaction_id="d-004", verdict=ReviewVerdict.CONFIDENCE_DOWNGRADED,
                                  confidence_adjustment=-0.05, reason="Thin evidence.")
    result = route_decision(score_result, review_result)
    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN
    assert result.final_score == 0.15


def test_downgraded_verdict_cannot_auto_approve_a_nonsafe_original_score():
    score_result = ScoreResult(transaction_id="regression-002", score=0.05, evidence=[],
                                model_version="random-forest-v1")
    review_result = ReviewResult(transaction_id="regression-002",
                                  verdict=ReviewVerdict.CONFIDENCE_DOWNGRADED,
                                  confidence_adjustment=-0.05, reason="Thin evidence.")
    result = route_decision(score_result, review_result)
    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN


def test_small_negative_adjustment_cannot_push_escalate_zone_to_approve():
    score_result = ScoreResult(transaction_id="regression-003", score=0.03, evidence=[],
                                model_version="random-forest-v1")
    review_result = ReviewResult(transaction_id="regression-003",
                                  verdict=ReviewVerdict.CONFIDENCE_UPHELD,
                                  confidence_adjustment=-0.02, reason="Minor concentration penalty.")
    result = route_decision(score_result, review_result)
    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN
