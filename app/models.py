from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class EvidenceDirection(str, Enum):
    SUPPORTS_FRAUD = "supports_fraud"
    CONTRADICTS_FRAUD = "contradicts_fraud"


class Evidence(BaseModel):
    signal: str
    direction: EvidenceDirection
    strength: float = Field(ge=0.0, le=1.0)
    description: str


class TransactionIn(BaseModel):
    transaction_id: str
    card_id: str
    amount: float
    currency: str = "INR"
    merchant_id: str
    merchant_country: str
    customer_country: str
    timestamp: datetime


class FeatureVector(BaseModel):
    transaction_id: str
    velocity_1h: int
    refund_rate_30d: float
    geo_mismatch: bool
    settlement_delay_hours: float
    amount_zscore: float


class DriftResult(BaseModel):
    transaction_id: str
    drift_score: float = Field(ge=0.0, le=1.0)
    drift_signals: list[str]
    framing_note: str = "Defensive drift detection only. Not a model of fraud techniques."


class ScoreResult(BaseModel):
    transaction_id: str
    score: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence]
    model_version: str = "rule-based-v0"


class ReviewVerdict(str, Enum):
    CONFIDENCE_UPHELD = "confidence_upheld"
    CONFIDENCE_DOWNGRADED = "confidence_downgraded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReviewResult(BaseModel):
    transaction_id: str
    verdict: ReviewVerdict
    confidence_adjustment: float = Field(ge=-1.0, le=0.0)
    reason: str


class DecisionOutcome(str, Enum):
    AUTO_APPROVE = "auto_approve"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    AUTO_REJECT = "auto_reject"


class Decision(BaseModel):
    transaction_id: str
    outcome: DecisionOutcome
    final_score: float
    reason: str


class AuditEntry(BaseModel):
    transaction_id: str
    timestamp: datetime
    stage: str
    actor: str
    data: dict
