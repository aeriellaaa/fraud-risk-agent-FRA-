"""
Finds the highest-scoring real fraud cases from credit_card_fraud_2026.csv,
using the actual trained model -- not guessed. These become your curated
demo transactions: real data, real fraud labels, real high scores with
real supporting evidence, so a live demo isn't gambling on a case the
model happens to score low (a real, confirmed property of this model --
see debug-log.md).

Run with: python scripts/find_demo_fraud_cases.py
"""

import pandas as pd
from app.models import TransactionIn
from app.agents.scoring_agent import score_transaction

DATA_PATH = "data/credit_card_fraud_2026.csv"
TOP_N = 5

FEATURE_COLS = [
    "amount_usd", "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age", "account_balance_usd",
    "is_new_merchant", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
    "cvv_retry_count", "velocity_score", "time_of_day_hour", "day_of_week",
    "is_ai_generated_scam_attempt", "merchant_risk_score", "prior_disputes",
]


def main():
    df = pd.read_csv(DATA_PATH)
    fraud_rows = df[df["is_fraud"] == 1].copy()
    print(f"Scoring all {len(fraud_rows)} real fraud cases against the trained model...")

    results = []
    for _, row in fraud_rows.iterrows():
        row_dict = row[FEATURE_COLS].to_dict()
        row_dict["transaction_id"] = str(row["transaction_id"])
        txn = TransactionIn(**row_dict)
        score_result = score_transaction(txn)
        results.append((score_result.score, txn.transaction_id, score_result))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n=== TOP {TOP_N} HIGHEST-SCORING REAL FRAUD CASES (use these for the demo) ===\n")
    for score, txn_id, score_result in results[:TOP_N]:
        print(f"Transaction {txn_id} -- score {score:.4f}")
        for e in score_result.evidence:
            arrow = "supports" if e.direction.value == "supports_fraud" else "contradicts"
            print(f"  [{arrow}] {e.signal}: {e.description}")
        print()

    print(f"Score distribution across all {len(fraud_rows)} fraud cases:")
    scores = [r[0] for r in results]
    print(f"  max={max(scores):.4f}  min={min(scores):.4f}  "
          f"median={sorted(scores)[len(scores)//2]:.4f}")


if __name__ == "__main__":
    main()
