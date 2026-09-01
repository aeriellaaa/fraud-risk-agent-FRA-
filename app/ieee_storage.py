"""
Separate in-memory store for the IEEE pipeline, kept independent from the main store.py
so nothing about the tested, working original pipeline is touched by this migration work.
"""
from typing import Optional
from app.models import IEEETransactionIn, ScoreResult, ReviewResult, Decision, AuditEntry

class IEEEStore:
    def __init__(self):
        self.transactions: dict[str, IEEETransactionIn] = {}
        self.scores: dict[str, ScoreResult] = {}
        self.reviews: dict[str, ReviewResult] = {}
        self.decisions: dict[str, Decision] = {}
        self.audit_log: list[AuditEntry] = []

    def save_transaction(self, txn: IEEETransactionIn) -> None:
        self.transactions[txn.transaction_id] = txn

    def get_transaction(self, transaction_id: str) -> Optional[IEEETransactionIn]:
        return self.transactions.get(transaction_id)

    def save_score(self, score: ScoreResult) -> None:
        self.scores[score.transaction_id] = score

    def get_score(self, transaction_id: str) -> Optional[ScoreResult]:
        return self.scores.get(transaction_id)

    def save_review(self, review: ReviewResult) -> None:
        self.reviews[review.transaction_id] = review

    def get_review(self, transaction_id: str) -> Optional[ReviewResult]:
        return self.reviews.get(transaction_id)

    def save_decision(self, decision: Decision) -> None:
        self.decisions[decision.transaction_id] = decision

    def append_audit(self, entry: AuditEntry) -> None:
        self.audit_log.append(entry)

    def get_audit_log(self, transaction_id: Optional[str] = None) -> list[AuditEntry]:
        if transaction_id is None:
            return self.audit_log
        return [e for e in self.audit_log if e.transaction_id == transaction_id]

ieee_store = IEEEStore()
