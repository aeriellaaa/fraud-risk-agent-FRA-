import requests

BASE = "http://127.0.0.1:8000"

txn = {
    "transaction_id": "ieee_test_001",
    "TransactionAmt": 500.0,
    "ProductCD": "W",
    "card1": 13553, "card2": 225.0, "card4": "visa", "card6": "debit",
    "addr1": 299.0, "P_emaildomain": "gmail.com",
    "C13": 50.0, "C1": 10.0,
}

r = requests.post(f"{BASE}/ieee/transactions/ingest", json=txn)
print("Ingest:", r.status_code, r.json())

r = requests.post(f"{BASE}/ieee/transactions/{txn['transaction_id']}/process")
print("\nFull pipeline result:")
import json
print(json.dumps(r.json(), indent=2, default=str))
