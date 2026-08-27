from app.models import FeatureVector
from app.agents.scoring_agent import score_transaction
from app.models import DriftResult


def test_clean_transaction_gets_low_score():
    features = FeatureVector(
        transaction_id="test-clean-001",
        velocity_flag=False,
        geo_mismatch=False,
        device_risk=False,
        new_merchant_risk=False,
        cvv_risk=False,
        prior_dispute_risk=False,
        high_amount_ratio=False,
        merchant_risk_score=10.0,
        ai_scam_flag=False,
    )

    result = score_transaction(features)

    assert result.transaction_id == "test-clean-001"
    assert result.score == 0.0
    assert result.model_version == "rule-based-v0"
def test_risky_transaction_gets_supporting_evidence():
    features = FeatureVector(
        transaction_id="test-risky-001",
        velocity_flag=True,
        geo_mismatch=True,
        device_risk=True,
        new_merchant_risk=True,
        cvv_risk=True,
        prior_dispute_risk=True,
        high_amount_ratio=True,
        merchant_risk_score=60.0,
        ai_scam_flag=True,
    )

    result = score_transaction(features)

    assert result.transaction_id == "test-risky-001"
    assert result.score > 0.0

    signals = {e.signal for e in result.evidence}

    assert "geo_mismatch" in signals
    assert "cvv_risk" in signals
    assert "ai_scam_flag" in signals
    assert "velocity_flag" in signals
    assert "device_risk" in signals
    assert "merchant_risk_high" in signals
    assert "new_merchant_risk" in signals
    assert "prior_dispute_risk" in signals
    assert "high_amount_ratio" in signals

def test_drift_result_becomes_pattern_evasion_evidence():
    features = FeatureVector(
        transaction_id="test-drift-001",
        velocity_flag=False,
        geo_mismatch=False,
        device_risk=False,
        new_merchant_risk=False,
        cvv_risk=False,
        prior_dispute_risk=False,
        high_amount_ratio=False,
        merchant_risk_score=20.0,
        ai_scam_flag=False,
    )

    drift = DriftResult(
        transaction_id="test-drift-001",
        drift_score=0.8,
        drift_signals=["unusual_velocity_pattern"],
    )

    result = score_transaction(features, drift)

    pattern_evidence = [
        e for e in result.evidence
        if e.signal == "pattern_evasion"
    ]

    assert len(pattern_evidence) == 1
    assert pattern_evidence[0].direction.value == "supports_fraud"
    assert pattern_evidence[0].strength == 0.50 * 0.8
def test_clean_signals_create_contradicting_evidence():
    features = FeatureVector(
        transaction_id="test-contradict-001",
        velocity_flag=False,
        geo_mismatch=False,
        device_risk=False,
        new_merchant_risk=False,
        cvv_risk=False,
        prior_dispute_risk=False,
        high_amount_ratio=False,
        merchant_risk_score=10.0,
        ai_scam_flag=False,
    )

    result = score_transaction(features)

    contradictions = [
        e for e in result.evidence
        if e.direction.value == "contradicts_fraud"
    ]

    signals = {e.signal for e in contradictions}

    assert "merchant_risk_low" in signals
    assert "clean_device_and_location" in signals