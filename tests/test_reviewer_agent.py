from app.agents.reviewer_agent import review_score
from app.models import ScoreResult, ReviewVerdict
from app.models import Evidence, EvidenceDirection
def test_no_supporting_evidence_is_insufficient():
    score_result = ScoreResult(
        transaction_id="review-test-001",
        score=0.0,
        evidence=[],
        model_version="rule-based-v0",
    )

    result = review_score(score_result)

    assert result.transaction_id == "review-test-001"
    assert result.verdict == ReviewVerdict.INSUFFICIENT_EVIDENCE
    assert result.confidence_adjustment == 0.0
def test_high_score_with_thin_evidence_is_downgraded():
    score_result = ScoreResult(
        transaction_id="review-test-002",
        score=0.8,
        evidence=[
            Evidence(
                signal="geo_mismatch",
                direction=EvidenceDirection.SUPPORTS_FRAUD,
                strength=0.7,
                description="Location mismatch detected",
            )
        ],
        model_version="rule-based-v0",
    )

    result = review_score(score_result)

    assert result.verdict == ReviewVerdict.CONFIDENCE_DOWNGRADED
    assert result.confidence_adjustment == -0.35
    assert "thin evidence" in result.reason
def test_strong_diverse_evidence_is_upheld():
    score_result = ScoreResult(
        transaction_id="review-test-003",
        score=0.8,
        evidence=[
            Evidence(
                signal="geo_mismatch",
                direction=EvidenceDirection.SUPPORTS_FRAUD,
                strength=0.7,
                description="Location mismatch detected",
            ),
            Evidence(
                signal="velocity_flag",
                direction=EvidenceDirection.SUPPORTS_FRAUD,
                strength=0.45,
                description="Unusual transaction velocity",
            ),
            Evidence(
                signal="merchant_risk_high",
                direction=EvidenceDirection.SUPPORTS_FRAUD,
                strength=0.45,
                description="High merchant risk",
            ),
            Evidence(
                signal="prior_dispute_risk",
                direction=EvidenceDirection.SUPPORTS_FRAUD,
                strength=0.30,
                description="Prior disputes detected",
            ),
        ],
        model_version="rule-based-v0",
    )

    result = review_score(score_result)

    assert result.verdict == ReviewVerdict.CONFIDENCE_UPHELD
    assert result.confidence_adjustment == 0.0
def test_strong_contradicting_evidence_reduces_confidence():
    score_result = ScoreResult(
        transaction_id="review-test-004",
        score=0.7,
        evidence=[
            Evidence(
                signal="geo_mismatch",
                direction=EvidenceDirection.SUPPORTS_FRAUD,
                strength=0.7,
                description="Location mismatch detected",
            ),
            Evidence(
                signal="velocity_flag",
                direction=EvidenceDirection.SUPPORTS_FRAUD,
                strength=0.45,
                description="Unusual transaction velocity",
            ),
            Evidence(
                signal="merchant_risk_low",
                direction=EvidenceDirection.CONTRADICTS_FRAUD,
                strength=0.6,
                description="Merchant risk is low",
            ),
        ],
        model_version="rule-based-v0",
    )

    result = review_score(score_result)

    assert result.verdict == ReviewVerdict.CONFIDENCE_DOWNGRADED
    assert result.confidence_adjustment == -0.30
    assert "contradicting signal" in result.reason