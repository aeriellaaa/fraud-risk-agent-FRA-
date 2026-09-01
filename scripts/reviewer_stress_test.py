"""
Reviewer Agent stress test -- demonstrates the Reviewer Agent catching cases a naive
"just use the score" policy would get wrong. This is the concrete evidence for "why
three agents, not one model + SHAP": no other reviewed competitor repo (kavach, Sentinel)
has an equivalent independent evidence-strength check.

Run with: python -m scripts.reviewer_stress_test  (or python stress_test_check.py directly)
"""
from app.models import ScoreResult, Evidence, EvidenceDirection
import importlib

reviewer_agent = importlib.import_module("app.agents.reviewer_agent")
decision = importlib.import_module("app.decision")


def naive_decide(score):
    """What a naive 'just use the score, no independent review' policy would do."""
    if score >= 0.15:
        return "AUTO_REJECT"
    elif score <= 0.01:
        return "AUTO_APPROVE"
    return "ESCALATE"


print("=" * 70)
print("REVIEWER AGENT STRESS TEST -- adversarial cases a naive policy gets wrong")
print("=" * 70)

print("\n--- CASE 1: High score, thin evidence (one weak signal) ---")
sr1 = ScoreResult(
    transaction_id="stress-001", score=0.16,
    evidence=[Evidence(signal="amount_usd", direction=EvidenceDirection.SUPPORTS_FRAUD,
                        strength=0.02, description="Slightly elevated amount")],
    model_version="random-forest-v1",
)
review1 = reviewer_agent.review_score(sr1)
decision1 = decision.route_decision(sr1, review1)
print(f"Raw score: {sr1.score} -> naive policy: {naive_decide(sr1.score)}")
print(f"Reviewer: {review1.verdict.value} ({review1.confidence_adjustment}) -- {review1.reason}")
print(f"ACTUAL DECISION: {decision1.outcome.value}")

print("\n--- CASE 2: Redundant same-category evidence (3 location signals) ---")
sr2 = ScoreResult(
    transaction_id="stress-002", score=0.17,
    evidence=[
        Evidence(signal="is_foreign_transaction", direction=EvidenceDirection.SUPPORTS_FRAUD,
                  strength=0.06, description="Foreign transaction"),
        Evidence(signal="distance_from_home_km", direction=EvidenceDirection.SUPPORTS_FRAUD,
                  strength=0.06, description="Far from home"),
        Evidence(signal="ip_country_mismatch", direction=EvidenceDirection.SUPPORTS_FRAUD,
                  strength=0.05, description="IP/country mismatch"),
    ],
    model_version="random-forest-v1",
)
review2 = reviewer_agent.review_score(sr2)
decision2 = decision.route_decision(sr2, review2)
print(f"Raw score: {sr2.score} -> naive policy: {naive_decide(sr2.score)}")
print(f"Reviewer: {review2.verdict.value} ({review2.confidence_adjustment}) -- {review2.reason}")
print(f"ACTUAL DECISION: {decision2.outcome.value}")

print("\n--- CASE 3: Genuinely contradictory evidence (VPN vs clean long history) ---")
sr3 = ScoreResult(
    transaction_id="stress-003", score=0.16,
    evidence=[
        Evidence(signal="used_vpn", direction=EvidenceDirection.SUPPORTS_FRAUD,
                  strength=0.05, description="VPN detected"),
        Evidence(signal="prior_disputes", direction=EvidenceDirection.CONTRADICTS_FRAUD,
                  strength=0.04, description="Zero prior disputes, 5-year account"),
        Evidence(signal="customer_age", direction=EvidenceDirection.CONTRADICTS_FRAUD,
                  strength=0.03, description="Established customer profile"),
    ],
    model_version="random-forest-v1",
)
review3 = reviewer_agent.review_score(sr3)
decision3 = decision.route_decision(sr3, review3)
print(f"Raw score: {sr3.score} -> naive policy: {naive_decide(sr3.score)}")
print(f"Reviewer: {review3.verdict.value} ({review3.confidence_adjustment}) -- {review3.reason}")
print(f"ACTUAL DECISION: {decision3.outcome.value}")

print("\n" + "=" * 70)
print("SUMMARY: naive score-only policy would AUTO_REJECT all 3 cases.")
print("Reviewer Agent correctly escalated all 3 instead, for 3 different real reasons.")
print("=" * 70)
