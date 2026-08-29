"""
End-to-end pipeline sanity check.

Run with: python -m pytest tests/test_pipeline_e2e.py -v -s

Uses two real rows from credit_card_fraud_2026.csv:
  - transaction 64: labeled fraud, but ambiguous (no single overwhelming
    signal -- tests whether the system reasons about combinations, not
    just one red flag)
  - transaction 1: labeled clean, unremarkable -- tests that the system
    doesn't over-trigger on a normal transaction
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

FRAUD_TXN = {
    "transaction_id": "test-64",
    "amount_usd": 2396.82,
    "merchant_category": "Crypto Exchange",
    "card_type": "Visa",
    "auth_method": "OTP",
    "channel": "ATM",
    "device_type": "Mac",
    "is_foreign_transaction": False,
    "hours_since_last_txn": 5.37,
    "txn_count_last_24h": 3,
    "distance_from_home_km": 15.8,
    "card_age_months": 46,
    "customer_age": 23,
    "account_balance_usd": 2611.03,
    "is_new_merchant": True,
    "used_vpn": False,
    "ip_country_mismatch": False,
    "billing_shipping_mismatch": False,
    "cvv_retry_count": 1,
    "velocity_score": 11.3,
    "time_of_day_hour": 5,
    "day_of_week": 5,
    "is_ai_generated_scam_attempt": False,
    "merchant_risk_score": 92.0,
    "prior_disputes": 0,
}

CLEAN_TXN = {
    "transaction_id": "test-1",
    "amount_usd": 42.86,
    "merchant_category": "Restaurants",
    "card_type": "Visa",
    "auth_method": "OTP",
    "channel": "Online",
    "device_type": "Android Phone",
    "is_foreign_transaction": False,
    "hours_since_last_txn": 13.54,
    "txn_count_last_24h": 2,
    "distance_from_home_km": 22.35,
    "card_age_months": 39,
    "customer_age": 25,
    "account_balance_usd": 1080.71,
    "is_new_merchant": False,
    "used_vpn": False,
    "ip_country_mismatch": False,
    "billing_shipping_mismatch": False,
    "cvv_retry_count": 0,
    "velocity_score": 0.1,
    "time_of_day_hour": 18,
    "day_of_week": 3,
    "is_ai_generated_scam_attempt": False,
    "merchant_risk_score": 42.3,
    "prior_disputes": 0,
}


def run_pipeline(txn: dict) -> dict:
    r = client.post("/transactions/ingest", json=txn)
    assert r.status_code == 200, f"ingest failed: {r.text}"

    r = client.post(f"/transactions/{txn['transaction_id']}/process")
    assert r.status_code == 200, f"process failed: {r.text}"
    return r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_fraud_transaction_escalates_or_rejects():
    result = run_pipeline(FRAUD_TXN)
    decision = result["decision"]
    print(f"\nFraud txn -> outcome={decision['outcome']}, score={decision['final_score']}")
    print(f"  Agent 1 drift signals: {result['drift']['drift_signals']}")
    print(f"  Agent 2 evidence count: {len(result['score']['evidence'])}")
    print(f"  Agent 3 verdict: {result['review']['verdict']}")
    assert decision["outcome"] != "auto_approve", (
        f"Fraud-labeled transaction auto-approved -- this would be a real problem, "
        f"got: {decision}"
    )


def test_clean_transaction_does_not_reject():
    result = run_pipeline(CLEAN_TXN)
    decision = result["decision"]
    print(f"\nClean txn -> outcome={decision['outcome']}, score={decision['final_score']}")
    print(f"  Agent 2 evidence count: {len(result['score']['evidence'])}")
    print(f"  Agent 3 verdict: {result['review']['verdict']}")
    assert decision["outcome"] != "auto_reject", (
        f"Clean transaction auto-rejected -- this would be a false positive "
        f"problem, got: {decision}"
    )


def test_audit_log_captures_all_stages():
    r = client.get("/audit-log", params={"transaction_id": "test-64"})
    assert r.status_code == 200
    stages = {e["stage"] for e in r.json()}
    expected = {"ingest", "pattern_agent", "scoring_agent", "reviewer_agent", "decision_router"}
    missing = expected - stages
    assert not missing, f"Audit log missing stages: {missing}"
    print(f"\nAudit log stages captured: {stages}")
