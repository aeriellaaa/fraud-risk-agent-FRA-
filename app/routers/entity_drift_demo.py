"""
Demo-only router: exercises Agent 1's entity-drift mode against the
merged real-Razorpay-customer + simulated-features demo data.

Clearly separated from the main pipeline routers -- this endpoint exists
to DEMONSTRATE a capability, not to be part of the production-shaped
API surface. See scripts/build_entity_demo_data.py for what's real vs.
simulated in this data.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.models import TransactionIn
from app.agents.pattern_agent import detect_drift

router = APIRouter()

DEMO_FILE = Path("data/demo/merged_entity_demo.json")


def _load_demo_data() -> list[dict]:
    if not DEMO_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Demo data not found. Run scripts/generate_razorpay_data.py then "
                   "scripts/build_entity_demo_data.py first.",
        )
    with open(DEMO_FILE) as f:
        return json.load(f)


@router.get("/demo/entity-drift/customers")
def list_demo_customers():
    """Lists the real Razorpay customer IDs available for the entity-drift demo."""
    data = _load_demo_data()
    return [
        {
            "real_razorpay_customer_id": c["real_razorpay_customer_id"],
            "real_razorpay_customer_name": c["real_razorpay_customer_name"],
            "transaction_count": len(c["transactions"]),
        }
        for c in data
    ]


@router.post("/demo/entity-drift/{customer_id}")
def run_entity_drift_demo(customer_id: str):
    """
    Runs Agent 1's entity-drift mode: compares this real customer's most
    recent transaction against their own transaction history (real
    Razorpay linkage, simulated behavioral feature values -- see
    scripts/build_entity_demo_data.py header for the full breakdown).
    """
    data = _load_demo_data()
    customer_entry = next(
        (c for c in data if c["real_razorpay_customer_id"] == customer_id), None
    )
    if customer_entry is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in demo data")

    transactions = [TransactionIn(**{k: v for k, v in t.items() if not k.startswith("_")})
                     for t in customer_entry["transactions"]]

    if len(transactions) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 transactions to demo entity drift")

    *history, latest = transactions
    result = detect_drift(latest, card_history=history)

    return {
        "real_razorpay_customer_id": customer_id,
        "real_razorpay_customer_name": customer_entry["real_razorpay_customer_name"],
        "history_size": len(history),
        "latest_transaction_id": latest.transaction_id,
        "drift_result": result,
        "note": "customer_id and transaction_id are real Razorpay test-mode data. "
                "Behavioral feature values (velocity_score, etc.) are simulated -- "
                "Razorpay's Orders API does not track these.",
    }
