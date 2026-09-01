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
    amount_usd: float
    merchant_category: str
    card_type: str
    auth_method: str
    channel: str
    device_type: str
    is_foreign_transaction: bool
    hours_since_last_txn: float
    txn_count_last_24h: int
    distance_from_home_km: float
    card_age_months: int
    customer_age: int
    account_balance_usd: float
    is_new_merchant: bool
    used_vpn: bool
    ip_country_mismatch: bool
    billing_shipping_mismatch: bool
    cvv_retry_count: int
    velocity_score: float
    time_of_day_hour: int
    day_of_week: int
    is_ai_generated_scam_attempt: bool
    merchant_risk_score: float
    prior_disputes: int


class FeatureVector(BaseModel):
    transaction_id: str
    velocity_flag: bool
    geo_mismatch: bool
    device_risk: bool
    new_merchant_risk: bool
    cvv_risk: bool
    prior_dispute_risk: bool
    high_amount_ratio: bool
    merchant_risk_score: float
    ai_scam_flag: bool


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
"""
IEEE-CIS transaction schema. Raw, checkout-time-known fields are required. Aggregated/
historical fields (C1-C14 velocity counters, D1-D15 time-deltas, id_* device-fingerprint
signals) are optional -- a real production system would compute these from a transaction-
history lookup (same pattern as this repo's existing FeatureVector), but that pipeline
isn't built yet, so callers may omit them and the model falls back to training-set medians
(numeric) or "unknown" (categorical). This is a documented simplification, not a hidden one.
"""
from pydantic import BaseModel
from typing import Optional

class IEEETransactionIn(BaseModel):
    transaction_id: str
    TransactionAmt: float
    ProductCD: str
    card1: float
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    # Aggregated/historical signals -- optional, filled from training medians if absent
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D4: Optional[float] = None
    D10: Optional[float] = None
    D15: Optional[float] = None
    M4: Optional[str] = None
