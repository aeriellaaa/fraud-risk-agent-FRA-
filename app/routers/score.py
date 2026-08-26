from fastapi import APIRouter, HTTPException
from app.storage import store
from app.agents.pattern_agent import detect_drift
from app.agents.scoring_agent import score_transaction
from app.audit import log_stage

router = APIRouter()


@router.post("/transactions/{transaction_id}/score")
def score_endpoint(transaction_id: str):
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

    return {"drift": drift, "score": score}
