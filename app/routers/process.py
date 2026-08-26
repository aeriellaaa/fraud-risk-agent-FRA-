from fastapi import APIRouter, HTTPException
from app.storage import store
from app.agents.pattern_agent import detect_drift
from app.agents.scoring_agent import score_transaction
from app.agents.reviewer_agent import review_score
from app.decision import route_decision
from app.audit import log_stage

router = APIRouter()


@router.post("/transactions/{transaction_id}/process")
def process_full_pipeline(transaction_id: str):
    """
    Convenience endpoint: runs ingest-through-decision in one call.
    Transaction must already be ingested. Useful for testing and for
    the demo, so the full loop is one API call instead of four.
    """
    txn = store.get_transaction(transaction_id)
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found -- ingest it first")

    drift = detect_drift(txn)
    store.save_drift(drift)
    log_stage(transaction_id, "pattern_agent", "agent_1", {"drift": drift.model_dump()})

    features = store.get_features(transaction_id)
    score = score_transaction(features, drift=drift)
    store.save_score(score)
    log_stage(transaction_id, "scoring_agent", "agent_2", {"score": score.model_dump()})

    review = review_score(score)
    store.save_review(review)
    log_stage(transaction_id, "reviewer_agent", "agent_3", {"review": review.model_dump()})

    decision = route_decision(score, review)
    store.save_decision(decision)
    log_stage(transaction_id, "decision_router", "system", {"decision": decision.model_dump()})

    return {"drift": drift, "score": score, "review": review, "decision": decision}
