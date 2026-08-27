from app.models import TransactionIn, DecisionOutcome
from app.routers.ingest import ingest_transaction
from app.routers.process import process_full_pipeline


def test_full_pipeline_runs_from_transaction_to_decision():
    transaction = TransactionIn(
        transaction_id="pipeline-test-001",
        amount_usd=50.0,
        merchant_category="retail",
        card_type="credit",
        auth_method="chip",
        channel="pos",
        device_type="mobile",
        is_foreign_transaction=False,
        hours_since_last_txn=5.0,
        txn_count_last_24h=1,
        distance_from_home_km=5.0,
        card_age_months=24,
        customer_age=30,
        account_balance_usd=5000.0,
        is_new_merchant=False,
        used_vpn=False,
        ip_country_mismatch=False,
        billing_shipping_mismatch=False,
        cvv_retry_count=0,
        velocity_score=10.0,
        time_of_day_hour=14,
        day_of_week=2,
        is_ai_generated_scam_attempt=False,
        merchant_risk_score=10.0,
        prior_disputes=0,
    )

    ingest_result = ingest_transaction(transaction)

    assert ingest_result["status"] == "ingested"

    result = process_full_pipeline("pipeline-test-001")

    assert result["drift"].transaction_id == "pipeline-test-001"
    assert result["score"].transaction_id == "pipeline-test-001"
    assert result["review"].transaction_id == "pipeline-test-001"
    assert result["decision"].transaction_id == "pipeline-test-001"

    assert result["decision"].outcome == DecisionOutcome.AUTO_APPROVE