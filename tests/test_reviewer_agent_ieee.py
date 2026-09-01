"""
Tests for Agent 3 (IEEE Reviewer). Written against real thresholds in
reviewer_agent_ieee.py: HIGH_SCORE_THRESHOLD=0.3, DOWNGRADE_THRESHOLD=-0.08.
"""
from app.agents.reviewer_agent_ieee import review_score_ieee
from app.models import ScoreResult, ReviewVerdict, Evidence, EvidenceDirection


def test_no_supporting_evidence_is_insufficient():
    score_result = ScoreResult(
        transaction_id="ieee-review-001", score=0.0, evidence=[],
        model_version="lightgbm-ieee-v3",
    )
    result = review_score_ieee(score_result)
    assert result.verdict == ReviewVerdict.INSUFFICIENT_EVIDENCE
    assert result.confidence_adjustment == 0.0
    assert "No supporting evidence" in result.reason


def test_high_score_with_thin_evidence_is_downgraded():
    score_result = ScoreResult(
        transaction_id="ieee-review-002", score=0.5,
        evidence=[Evidence(signal="TransactionAmt", direction=EvidenceDirection.SUPPORTS_FRAUD,
                            strength=0.6, description="amount")],
        model_version="lightgbm-ieee-v3",
    )
    result = review_score_ieee(score_result)
    assert result.verdict == ReviewVerdict.CONFIDENCE_DOWNGRADED
    assert result.confidence_adjustment <= -0.08
    assert "thin evidence" in result.reason


def test_strong_diverse_evidence_is_upheld():
    score_result = ScoreResult(
        transaction_id="ieee-review-003", score=0.6,
        evidence=[
            Evidence(signal="TransactionAmt", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.3, description="financial"),
            Evidence(signal="card1", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.3, description="card"),
            Evidence(signal="C13", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.3, description="velocity_agg"),
            Evidence(signal="addr1", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.3, description="location"),
        ],
        model_version="lightgbm-ieee-v3",
    )
    result = review_score_ieee(score_result)
    assert result.verdict == ReviewVerdict.CONFIDENCE_UPHELD
    assert result.confidence_adjustment == 0.0


def test_evidence_concentrated_in_one_category_is_flagged():
    score_result = ScoreResult(
        transaction_id="ieee-review-004", score=0.6,
        evidence=[
            Evidence(signal="C1", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.3, description="velocity_agg"),
            Evidence(signal="C13", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.3, description="velocity_agg"),
            Evidence(signal="C6", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.3, description="velocity_agg"),
        ],
        model_version="lightgbm-ieee-v3",
    )
    result = review_score_ieee(score_result)
    assert "corroboration" in result.reason or result.confidence_adjustment < 0.0


def test_contradicting_evidence_reduces_confidence():
    score_result = ScoreResult(
        transaction_id="ieee-review-005", score=0.5,
        evidence=[
            Evidence(signal="TransactionAmt", direction=EvidenceDirection.SUPPORTS_FRAUD,
                      strength=0.4, description="financial"),
            Evidence(signal="card1", direction=EvidenceDirection.CONTRADICTS_FRAUD,
                      strength=0.3, description="card"),
        ],
        model_version="lightgbm-ieee-v3",
    )
    result = review_score_ieee(score_result)
    assert result.confidence_adjustment < 0.0
    assert "contradicting" in result.reason
