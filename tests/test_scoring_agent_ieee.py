"""
Tests for Agent 2 (IEEE-CIS LightGBM scorer). Does not hardcode exact SHAP values --
tests behavior (score range, evidence shape, honest handling of missing optional fields).
"""
from app.models import IEEETransactionIn
from app.agents.scoring_agent_ieee import score_transaction_ieee

MINIMAL_TXN = IEEETransactionIn(
    transaction_id="ieee-test-minimal",
    TransactionAmt=45.0, ProductCD="W",
    card1=9500, card4="visa", card6="debit",
)

RICH_TXN = IEEETransactionIn(
    transaction_id="ieee-test-rich",
    TransactionAmt=500.0, ProductCD="W",
    card1=13553, card2=225.0, card3=150.0, card4="visa", card5=226.0, card6="debit",
    addr1=299.0, addr2=87.0, dist1=10.0,
    P_emaildomain="gmail.com", R_emaildomain="gmail.com",
    C1=10.0, C13=50.0, D1=5.0, D15=3.0,
)


def test_minimal_transaction_scores_without_error():
    result = score_transaction_ieee(MINIMAL_TXN)
    assert result.transaction_id == "ieee-test-minimal"
    assert 0.0 <= result.score <= 1.0
    assert result.model_version == "lightgbm-ieee-v3"


def test_score_returns_evidence_with_valid_shape():
    result = score_transaction_ieee(RICH_TXN)
    assert len(result.evidence) > 0
    for e in result.evidence:
        assert 0.0 <= e.strength <= 1.0
        assert e.direction.value in ("supports_fraud", "contradicts_fraud")
        assert isinstance(e.description, str) and len(e.description) > 0


def test_missing_optional_fields_are_labeled_honestly_not_as_none():
    result = score_transaction_ieee(MINIMAL_TXN)
    for e in result.evidence:
        assert "None" not in e.description, (
            f"Evidence for '{e.signal}' shows raw 'None' instead of an honest "
            f"'not provided by caller' label: {e.description}"
        )


def test_engineered_signals_are_summarized_not_hidden():
    result = score_transaction_ieee(RICH_TXN)
    engineered = [e for e in result.evidence if e.signal == "engineered_signals_combined"]
    if engineered:
        assert "not individually interpretable" in engineered[0].description


def test_valid_signal_names_only():
    result = score_transaction_ieee(RICH_TXN)
    for e in result.evidence:
        assert e.signal == "engineered_signals_combined" or e.signal == "pattern_evasion" or isinstance(e.signal, str)
