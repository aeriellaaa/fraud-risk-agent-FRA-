"""
Agent 1 -- Pattern/Evasion Agent.

IMPORTANT (defensive framing, non-negotiable): this agent performs
DEFENSIVE DRIFT AND EVASION DETECTION ONLY. It does not model, simulate,
or generate fraud techniques.

Two modes, chosen automatically based on available data:

1. PER-ENTITY HISTORICAL DRIFT (when card_history is provided): compares
   a transaction's key signals against that specific entity's own
   historical average. Demonstrated against real Razorpay customer
   linkage with simulated behavioral features -- see
   scripts/build_entity_demo_data.py and app/routers/entity_drift_demo.py.

2. POPULATION-LEVEL FALLBACK (when no history is available): multivariate
   anomaly + threshold-evasion detection against dataset-wide baselines.
   This is what applies to a single, ungrouped transaction -- the main
   production pipeline's default mode.
"""

from app.models import TransactionIn, DriftResult

BASELINE = {
    "velocity_score": (19.81, 12.37),
    "txn_count_last_24h": (3.19, 1.78),
    "merchant_risk_score": (37.40, 17.06),
    "cvv_retry_count": (0.18, 0.42),
}

Z_SCORE_ELEVATED = 1.5
MULTIVARIATE_MIN_SIGNALS = 2

VELOCITY_THRESHOLD = 27.6
MERCHANT_RISK_THRESHOLD = 47.3
EVASION_MARGIN_VELOCITY = 3.0
EVASION_MARGIN_MERCHANT_RISK = 5.0

ENTITY_DRIFT_Z_THRESHOLD = 1.2


def _zscore(value: float, key: str) -> float:
    mean, std = BASELINE[key]
    return (value - mean) / std if std > 0 else 0.0


def detect_drift(txn: TransactionIn, card_history: list[TransactionIn] | None = None) -> DriftResult:
    if card_history and len(card_history) >= 3:
        return _detect_entity_drift(txn, card_history)
    return _detect_population_drift(txn)


def _detect_entity_drift(txn: TransactionIn, card_history: list[TransactionIn]) -> DriftResult:
    signals: list[str] = []

    tracked_fields = ["velocity_score", "merchant_risk_score", "cvv_retry_count", "amount_usd"]
    elevated = []
    for field in tracked_fields:
        history_values = [getattr(h, field) for h in card_history]
        hist_mean = sum(history_values) / len(history_values)
        hist_std = (sum((v - hist_mean) ** 2 for v in history_values) / len(history_values)) ** 0.5
        current_value = getattr(txn, field)
        if hist_std > 0:
            z = (current_value - hist_mean) / hist_std
            if z > ENTITY_DRIFT_Z_THRESHOLD:
                elevated.append(field)
                signals.append(
                    f"entity_drift: {field}={current_value:.2f} is {z:.1f} std devs above "
                    f"this card's own history (mean={hist_mean:.2f}, n={len(history_values)})"
                )

    drift_score = min(1.0, len(elevated) / len(tracked_fields)) if elevated else 0.0

    return DriftResult(
        transaction_id=txn.transaction_id,
        drift_score=round(drift_score, 4),
        drift_signals=signals,
    )


def _detect_population_drift(txn: TransactionIn) -> DriftResult:
    signals: list[str] = []

    elevated = []
    for key in ("velocity_score", "txn_count_last_24h", "merchant_risk_score", "cvv_retry_count"):
        value = getattr(txn, key)
        z = _zscore(value, key)
        if z > Z_SCORE_ELEVATED:
            elevated.append(key)

    if len(elevated) >= MULTIVARIATE_MIN_SIGNALS:
        signals.append(
            f"multivariate_anomaly: {len(elevated)} signals simultaneously elevated "
            f"({', '.join(elevated)})"
        )

    if VELOCITY_THRESHOLD - EVASION_MARGIN_VELOCITY <= txn.velocity_score < VELOCITY_THRESHOLD:
        signals.append(
            f"velocity_evasion: velocity_score {txn.velocity_score:.1f} sits just under "
            f"the {VELOCITY_THRESHOLD} scoring threshold"
        )

    if MERCHANT_RISK_THRESHOLD - EVASION_MARGIN_MERCHANT_RISK <= txn.merchant_risk_score < MERCHANT_RISK_THRESHOLD:
        signals.append(
            f"merchant_risk_evasion: merchant_risk_score {txn.merchant_risk_score:.1f} sits "
            f"just under the {MERCHANT_RISK_THRESHOLD} scoring threshold"
        )

    drift_score = 0.0
    if elevated:
        drift_score += 0.5 * min(1.0, len(elevated) / 3)
    evasion_hits = sum(1 for s in signals if "evasion" in s)
    if evasion_hits:
        drift_score += 0.5 * min(1.0, evasion_hits / 2)

    return DriftResult(
        transaction_id=txn.transaction_id,
        drift_score=round(min(1.0, drift_score), 4),
        drift_signals=signals,
    )
