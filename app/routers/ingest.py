from fastapi import APIRouter, HTTPException
from app.models import TransactionIn
from app.storage import store
from app.features import extract_features
from app.audit import log_stage

router = APIRouter()


@router.post("/transactions/ingest")
def ingest_transaction(txn: TransactionIn):
    store.save_transaction(txn)
    features = extract_features(txn)
    store.save_features(features)
    log_stage(txn.transaction_id, "ingest", "system", {"features": features.model_dump()})
    return {"transaction_id": txn.transaction_id, "status": "ingested", "features": features}
