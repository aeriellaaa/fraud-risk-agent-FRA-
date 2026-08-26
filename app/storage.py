from typing import Optional
from app.models import (
    TransactionIn, FeatureVector, DriftResult,
    ScoreResult, ReviewResult, Decision, AuditEntry
)


class InMemoryStore:
    """
    Phase 1 storage: everything lives in process memory.
    Swap this for a real DB (Postgres/SQLite) in Phase 2 -
    the interface (get/set methods) should stay the same
    so nothing upstream has to change.
    """
    def __init__(self):
        self.transactions: dict[str, TransactionIn] = {}
        self.features: dict[str, FeatureVector] = {}
        self.drift_results: dict[str, DriftResult] = {}
        self.scores: dict[str, ScoreResult] = {}
        self.reviews: dict[str, ReviewResult] = {}
        self.decisions: dict[str, Decision] = {}
        self.card_history: dict[str, list[TransactionIn]] = {}
        self.audit_log: list[AuditEntry] = []

    def save_transaction(self, txn: TransactionIn) -> None:
        self.transactions[txn.transaction_id] = txn
        self.card_history.setdefault(txn.card_id, []).append(txn)

    def get_transaction(self, transaction_id: str) -> Optional[TransactionIn]:
        return self.transactions.get(transaction_id)

    def get_card_history(self, card_id: str) -> list[TransactionIn]:
        return self.card_history.get(card_id, [])

    def save_features(self, features: FeatureVector) -> None:
        self.features[features.transaction_id] = features

    def get_features(self, transaction_id: str) -> Optional[FeatureVector]:
        return self.features.get(transaction_id)

    def save_drift(self, drift: DriftResult) -> None:
        self.drift_results[drift.transaction_id] = drift

    def get_drift(self, transaction_id: str) -> Optional[DriftResult]:
        return self.drift_results.get(transaction_id)

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

    def get_decision(self, transaction_id: str) -> Optional[Decision]:
        return self.decisions.get(transaction_id)

    def append_audit(self, entry: AuditEntry) -> None:
        # Append-only by convention: nothing in this class ever
        # deletes or mutates an existing audit_log entry.
        self.audit_log.append(entry)

    def get_audit_log(self, transaction_id: Optional[str] = None) -> list[AuditEntry]:
        if transaction_id is None:
            return self.audit_log
        return [e for e in self.audit_log if e.transaction_id == transaction_id]


# Single shared instance the whole app imports
store = InMemoryStore()
