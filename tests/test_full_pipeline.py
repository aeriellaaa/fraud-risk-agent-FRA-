"""
Full pipeline test, using a real dataset-matching transaction (same as
the manually-verified clean transaction 1 from live testing).
"""

from app.models import TransactionIn, DecisionOutcome
from app.routers.ingest import ingest_transaction
from app.routers.process import process_full_pipeline


def test_full_pipeline_runs_from_transaction_to_decision():
    transaction = TransactionIn(
        transaction_id="pipeline-test-001",
        amount_usd=42.86, merchant_category="Restaurants", card_type="Visa",
        auth_method="OTP", channel="Online", device_type="Android Phone",
        is_foreign_transaction=False, hours_since_last_txn=13.54, txn_count_last_24h=2,
        distance_from_home_km=22.35, card_age_months=39, customer_age=25,
        account_balance_usd=1080.71, is_new_merchant=False, used_vpn=False,
        ip_country_mismatch=False, billing_shipping_mismatch=False, cvv_retry_count=0,
        velocity_score=0.1, time_of_day_hour=18, day_of_week=3,
        is_ai_generated_scam_attempt=False, merchant_risk_score=42.3, prior_disputes=0,
    )

    ingest_result = ingest_transaction(transaction)
    assert ingest_result["status"] == "ingested"

    result = process_full_pipeline("pipeline-test-001")

    assert result["drift"].transaction_id == "pipeline-test-001"
    assert result["score"].transaction_id == "pipeline-test-001"
    assert result["review"].transaction_id == "pipeline-test-001"
    assert result["decision"].transaction_id == "pipeline-test-001"
    assert result["decision"].outcome == DecisionOutcome.AUTO_APPROVE
