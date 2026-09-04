"""
Tests for Agent 3 (Reviewer). Written against real SHAP-scale evidence
magnitudes (roughly 0.005-0.15) and the actual thresholds in
reviewer_agent.py: HIGH_SCORE_THRESHOLD=0.01, DOWNGRADE_THRESHOLD=-0.05.
"""

from app.agents.reviewer_agent import review_score
from app.models import ScoreResult, ReviewVerdict, Evidence, EvidenceDirection


def test_no_supporting_evidence_is_insufficient():
    score_result = ScoreResult(
        transaction_id="review-test-001", score=0.0, evidence=[],
        model_version="random-forest-v1",
    )
    result = review_score(score_result)
    assert result.verdict == ReviewVerdict.INSUFFICIENT_EVIDENCE
    assert result.confidence_adjustment == 0.0
    assert "No supporting evidence" in result.reason


def test_high_score_with_thin_evidence_is_downgraded():
    score_result = ScoreResult(
        transaction_id="review-test-002", score=0.05,
        evidence=[Evidence(signal="cvv_retry_count", direction=EvidenceDirection.SUPPORTS_FRAUD,
                            strength=0.07, description="CVV retry")],
        model_version="random-forest-v1",
    )
    result = review_score(score_result)
    assert result.verdict == ReviewVerdict.CONFIDENCE_DOWNGRADED
    assert result.confidence_adjustment <= -0.05
    assert "thin evidence" in result.reason


def test_strong_diverse_evidence_is_upheld():
    score_result = ScoreResult(
        transaction_id="review-test-003", score=0.06,
        evidence=[
            Evidence(signal="amount_usd", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.05, description="financial"),
            Evidence(signal="merchant_risk_score", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.05, description="merchant"),
            Evidence(signal="velocity_score", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.05, description="behavior"),
            Evidence(signal="cvv_retry_count", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.05, description="authentication"),
        ],
        model_version="random-forest-v1",
    )
    result = review_score(score_result)
    assert result.verdict == ReviewVerdict.CONFIDENCE_UPHELD
    assert result.confidence_adjustment == 0.0


def test_strong_contradicting_evidence_reduces_confidence():
    # 1 supporting signal (triggers "thin evidence", -0.05) plus 2
    # contradicting signals at 0.06 each (triggers contradiction check,
    # 0.12 * 0.3 = -0.036). Combined -0.086, clearly past the -0.05
    # downgrade threshold -- not a borderline case.
    score_result = ScoreResult(
        transaction_id="review-test-004", score=0.06,
        evidence=[
            Evidence(signal="amount_usd", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.03, description="financial"),
            Evidence(signal="velocity_score", direction=EvidenceDirection.CONTRADICTS_FRAUD,
                      strength=0.06, description="behavior"),
            Evidence(signal="customer_age", direction=EvidenceDirection.CONTRADICTS_FRAUD,
                      strength=0.06, description="profile"),
        ],
        model_version="random-forest-v1",
    )
    result = review_score(score_result)
    assert result.verdict == ReviewVerdict.CONFIDENCE_DOWNGRADED
    assert result.confidence_adjustment < -0.05
    assert "contradicting signal" in result.reason


def test_contradiction_penalty_is_relative_not_absolute():
    score_result = ScoreResult(
        transaction_id="regression-001", score=0.2333,
        evidence=[
            Evidence(signal="cvv_retry_count", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.1206, description="cvv"),
            Evidence(signal="ip_country_mismatch", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.0946, description="geo"),
            Evidence(signal="is_new_merchant", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.0515, description="merchant"),
            Evidence(signal="time_of_day_hour", direction=EvidenceDirection.CONTRADICTS_FRAUD,
                      strength=0.0812, description="time"),
            Evidence(signal="account_balance_usd", direction=EvidenceDirection.CONTRADICTS_FRAUD,
                      strength=0.0806, description="balance"),
            Evidence(signal="distance_from_home_km", direction=EvidenceDirection.CONTRADICTS_FRAUD,
                      strength=0.0594, description="distance"),
        ],
        model_version="random-forest-v1",
    )
    result = review_score(score_result)
    assert result.verdict == ReviewVerdict.CONFIDENCE_UPHELD
