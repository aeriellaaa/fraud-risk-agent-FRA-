"""
Tests for Agent 2 (ML scorer). Written against the real Random Forest +
SHAP implementation -- does not hardcode exact SHAP values, since those
can shift slightly across library versions. Tests behavior, not exact
numbers.
"""

from app.models import TransactionIn, DriftResult
from app.agents.scoring_agent import score_transaction

CLEAN_TXN = TransactionIn(
    transaction_id="test-clean-001",
    amount_usd=42.86, merchant_category="Restaurants", card_type="Visa",
    auth_method="OTP", channel="Online", device_type="Android Phone",
    is_foreign_transaction=False, hours_since_last_txn=13.54, txn_count_last_24h=2,
    distance_from_home_km=22.35, card_age_months=39, customer_age=25,
    account_balance_usd=1080.71, is_new_merchant=False, used_vpn=False,
    ip_country_mismatch=False, billing_shipping_mismatch=False, cvv_retry_count=0,
    velocity_score=0.1, time_of_day_hour=18, day_of_week=3,
    is_ai_generated_scam_attempt=False, merchant_risk_score=42.3, prior_disputes=0,
)

RISKY_TXN = TransactionIn(
    transaction_id="test-risky-001",
    amount_usd=2396.82, merchant_category="Crypto Exchange", card_type="Visa",
    auth_method="OTP", channel="ATM", device_type="Mac",
    is_foreign_transaction=False, hours_since_last_txn=5.37, txn_count_last_24h=3,
    distance_from_home_km=15.8, card_age_months=46, customer_age=23,
    account_balance_usd=2611.03, is_new_merchant=True, used_vpn=False,
    ip_country_mismatch=False, billing_shipping_mismatch=False, cvv_retry_count=1,
    velocity_score=11.3, time_of_day_hour=5, day_of_week=5,
    is_ai_generated_scam_attempt=False, merchant_risk_score=92.0, prior_disputes=0,
)


def test_clean_transaction_gets_low_score():
    result = score_transaction(CLEAN_TXN)
    assert result.transaction_id == "test-clean-001"
    assert 0.0 <= result.score < 0.05
    assert result.model_version == "random-forest-v1"


def test_score_returns_evidence_with_valid_shape():
    result = score_transaction(RISKY_TXN)
    assert len(result.evidence) > 0
    for e in result.evidence:
        assert 0.0 <= e.strength <= 1.0
        assert e.direction.value in ("supports_fraud", "contradicts_fraud")
        assert isinstance(e.description, str) and len(e.description) > 0


def test_evidence_signals_come_from_real_features():
    result = score_transaction(RISKY_TXN)
    valid_signals = set(TransactionIn.model_fields.keys()) | {"pattern_evasion"}
    for e in result.evidence:
        assert e.signal in valid_signals, f"Unexpected signal: {e.signal}"


def test_drift_result_becomes_pattern_evasion_evidence():
    drift = DriftResult(
        transaction_id="test-drift-001",
        drift_score=0.8,
        drift_signals=["unusual_velocity_pattern"],
    )
    result = score_transaction(CLEAN_TXN, drift=drift)
    pattern_evidence = [e for e in result.evidence if e.signal == "pattern_evasion"]
    assert len(pattern_evidence) == 1
    assert pattern_evidence[0].direction.value == "supports_fraud"
    assert pattern_evidence[0].strength == round(min(1.0, 0.8 * 0.1), 4)


def test_low_drift_score_does_not_add_pattern_evasion_evidence():
    drift = DriftResult(transaction_id="test-drift-002", drift_score=0.1, drift_signals=[])
    result = score_transaction(CLEAN_TXN, drift=drift)
    pattern_evidence = [e for e in result.evidence if e.signal == "pattern_evasion"]
    assert len(pattern_evidence) == 0
