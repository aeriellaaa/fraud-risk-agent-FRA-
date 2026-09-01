"""
Tests for the IEEE Decision Router. Written against real thresholds:
AUTO_APPROVE_THRESHOLD=0.12, AUTO_REJECT_THRESHOLD=0.7574 (2% FP-budget operating
point -- NOT pure cost-optimal, see decision_ieee.py docstring), MARGIN=0.02.
"""
from app.decision_ieee import route_decision_ieee
from app.models import ScoreResult, ReviewResult, ReviewVerdict, DecisionOutcome


def test_low_score_is_auto_approved():
    score_result = ScoreResult(transaction_id="ieee-d-001", score=0.05, evidence=[],
                                model_version="lightgbm-ieee-v3")
    review_result = ReviewResult(transaction_id="ieee-d-001", verdict=ReviewVerdict.CONFIDENCE_UPHELD,
                                  confidence_adjustment=0.0, reason="No red flags.")
    result = route_decision_ieee(score_result, review_result)
    assert result.outcome == DecisionOutcome.AUTO_APPROVE


def test_high_trusted_score_is_auto_rejected():
    score_result = ScoreResult(transaction_id="ieee-d-002", score=0.9, evidence=[],
                                model_version="lightgbm-ieee-v3")
    review_result = ReviewResult(transaction_id="ieee-d-002", verdict=ReviewVerdict.CONFIDENCE_UPHELD,
                                  confidence_adjustment=0.0, reason="Strong evidence.")
    result = route_decision_ieee(score_result, review_result)
    assert result.outcome == DecisionOutcome.AUTO_REJECT
    assert result.final_score == 0.9


def test_middle_score_is_escalated():
    score_result = ScoreResult(transaction_id="ieee-d-003", score=0.5, evidence=[],
                                model_version="lightgbm-ieee-v3")
    review_result = ReviewResult(transaction_id="ieee-d-003", verdict=ReviewVerdict.CONFIDENCE_UPHELD,
                                  confidence_adjustment=0.0, reason="Evidence supports the score.")
    result = route_decision_ieee(score_result, review_result)
    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN


def test_downgraded_high_score_cannot_auto_reject():
    score_result = ScoreResult(transaction_id="ieee-d-004", score=0.9, evidence=[],
                                model_version="lightgbm-ieee-v3")
    review_result = ReviewResult(transaction_id="ieee-d-004", verdict=ReviewVerdict.CONFIDENCE_DOWNGRADED,
                                  confidence_adjustment=-0.3, reason="Thin evidence.")
    result = route_decision_ieee(score_result, review_result)
    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN
    assert result.final_score == 0.6


def test_score_near_reject_threshold_is_escalated_not_rejected():
    score_result = ScoreResult(transaction_id="ieee-d-005", score=0.75, evidence=[],
                                model_version="lightgbm-ieee-v3")
    review_result = ReviewResult(transaction_id="ieee-d-005", verdict=ReviewVerdict.CONFIDENCE_UPHELD,
                                  confidence_adjustment=0.0, reason="Borderline.")
    result = route_decision_ieee(score_result, review_result)
    assert result.outcome == DecisionOutcome.ESCALATE_TO_HUMAN
    assert "borderline margin" in result.reason
