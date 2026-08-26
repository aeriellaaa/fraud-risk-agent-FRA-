from fastapi import APIRouter
from app.storage import store
from app.models import DecisionOutcome

router = APIRouter()


@router.get("/metrics")
def get_metrics():
    """
    Phase 1 placeholder: basic decision-outcome counts from this session.
    Phase 2 replaces this with real precision/recall + false-positive
    cost analysis against a held-out labeled set -- the non-negotiable,
    not this placeholder.
    """
    decisions = list(store.decisions.values())
    counts = {outcome.value: 0 for outcome in DecisionOutcome}
    for d in decisions:
        counts[d.outcome.value] += 1
    return {
        "total_decisions": len(decisions),
        "breakdown": counts,
        "note": "Placeholder metrics. Real precision/recall + FP cost analysis is a Phase 2 non-negotiable.",
    }
