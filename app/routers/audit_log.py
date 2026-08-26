from fastapi import APIRouter, Query
from app.audit import read_audit_log

router = APIRouter()


@router.get("/audit-log")
def get_audit_log(transaction_id: str | None = Query(default=None)):
    return read_audit_log(transaction_id)
