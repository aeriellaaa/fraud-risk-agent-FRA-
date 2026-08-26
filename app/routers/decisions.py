from fastapi import APIRouter, HTTPException
from app.storage import store
from app.decision import route_decision
from app.audit import log_stage

router = APIRouter()


@router.post("/decisions/{transaction_id}")
def decide_endpoint(transaction_id: str):
    score = store.get_score(transaction_id)
    review = store.get_review(transaction_id)
    if score is None or review is None:
        raise HTTPException(status_code=404, detail="Score and review must both exist before deciding")

    decision = route_decision(score, review)
    store.save_decision(decision)
    log_stage(transaction_id, "decision_router", "system", {"decision": decision.model_dump()})

    return decision
