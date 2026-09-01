"""
IEEE pipeline routes, fully separate from the main pipeline -- runs alongside it under
the /ieee prefix so both can be demoed/compared without any risk to the tested main system.
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException

from app.models import IEEETransactionIn, AuditEntry
from app.ieee_storage import ieee_store
from app.agents.scoring_agent_ieee import score_transaction_ieee
from app.agents.reviewer_agent_ieee import review_score_ieee
from app.decision_ieee import route_decision_ieee

router = APIRouter(prefix="/ieee")


def _log_stage(transaction_id: str, stage: str, actor: str, data: dict) -> None:
    ieee_store.append_audit(AuditEntry(
        transaction_id=transaction_id,
        timestamp=datetime.utcnow(),
        stage=stage,
        actor=actor,
        data=data,
    ))


@router.post("/transactions/ingest")
def ingest_ieee(txn: IEEETransactionIn):
    ieee_store.save_transaction(txn)
    _log_stage(txn.transaction_id, "ingest", "system", {"transaction": txn.model_dump()})
    return {"status": "ingested", "transaction_id": txn.transaction_id}


@router.post("/transactions/{transaction_id}/score")
def score_ieee(transaction_id: str):
    txn = ieee_store.get_transaction(transaction_id)
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found -- ingest it first")
    score = score_transaction_ieee(txn)
    ieee_store.save_score(score)
    _log_stage(transaction_id, "scoring_agent_ieee", "agent_2", {"score": score.model_dump()})
    return score


@router.post("/transactions/{transaction_id}/review")
def review_ieee(transaction_id: str):
    score = ieee_store.get_score(transaction_id)
    if score is None:
        raise HTTPException(status_code=404, detail="No score found -- run /score first")
    review = review_score_ieee(score)
    ieee_store.save_review(review)
    _log_stage(transaction_id, "reviewer_agent_ieee", "agent_3", {"review": review.model_dump()})
    return review


@router.post("/decisions/{transaction_id}")
def decide_ieee(transaction_id: str):
    score = ieee_store.get_score(transaction_id)
    review = ieee_store.get_review(transaction_id)
    if score is None or review is None:
        raise HTTPException(status_code=404, detail="Score and review must both exist before deciding")
    decision = route_decision_ieee(score, review)
    ieee_store.save_decision(decision)
    _log_stage(transaction_id, "decision_router_ieee", "system", {"decision": decision.model_dump()})
    return decision


@router.post("/transactions/{transaction_id}/process")
def process_ieee(transaction_id: str):
    """Convenience endpoint: runs score -> review -> decide in one call."""
    txn = ieee_store.get_transaction(transaction_id)
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found -- ingest it first")

    score = score_transaction_ieee(txn)
    ieee_store.save_score(score)
    _log_stage(transaction_id, "scoring_agent_ieee", "agent_2", {"score": score.model_dump()})

    review = review_score_ieee(score)
    ieee_store.save_review(review)
    _log_stage(transaction_id, "reviewer_agent_ieee", "agent_3", {"review": review.model_dump()})

    decision = route_decision_ieee(score, review)
    ieee_store.save_decision(decision)
    _log_stage(transaction_id, "decision_router_ieee", "system", {"decision": decision.model_dump()})

    return {"score": score, "review": review, "decision": decision}


@router.get("/audit-log")
def audit_log_ieee(transaction_id: str | None = None):
    return ieee_store.get_audit_log(transaction_id)
