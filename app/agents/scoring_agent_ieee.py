"""
Agent 2 (IEEE-CIS version) -- LightGBM scoring using the V3 model. Evidence follows the
same pattern as scoring_agent.py: raw absolute SHAP contribution as strength, named
features shown individually, engineered/anonymized signals honestly summarized.

FIX from previous version: evidence descriptions now clearly label fields the caller
didn't provide, instead of misleadingly printing "None" next to a real contribution
(the model actually used an imputed default, not a null value).
"""
import pickle
import shap
import numpy as np
import pandas as pd
from pathlib import Path

from app.models import IEEETransactionIn, ScoreResult, Evidence, EvidenceDirection, DriftResult

ARTIFACT_DIR = Path("app/ml_artifacts_ieee_v3")

with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
    _model = pickle.load(f)
with open(ARTIFACT_DIR / "encoders.pkl", "rb") as f:
    _encoders = pickle.load(f)
with open(ARTIFACT_DIR / "freq_maps.pkl", "rb") as f:
    _freq_maps = pickle.load(f)
with open(ARTIFACT_DIR / "top_features.pkl", "rb") as f:
    _feature_order = pickle.load(f)
with open(ARTIFACT_DIR / "feature_medians.pkl", "rb") as f:
    _feature_medians = pickle.load(f)

_explainer = shap.TreeExplainer(_model)

INTERPRETABLE_PREFIXES = ("card", "TransactionAmt", "addr", "dist", "P_email", "R_email",
                           "D1", "D2", "D3", "D4", "D5", "D6", "D8", "D9", "D10", "D11",
                           "D12", "D13", "D14", "D15", "C1", "C2", "C3", "C4", "C5", "C6",
                           "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                           "hour_of_day", "day_of_week", "has_identity_data", "DeviceType",
                           "ProductCD", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9")

def _is_interpretable(feature_name: str) -> bool:
    return any(feature_name.startswith(p) for p in INTERPRETABLE_PREFIXES) and not feature_name.startswith("id_")

TOP_N_EVIDENCE = 6
MIN_CONTRIBUTION = 0.02


def _encode_transaction(txn_dict: dict) -> pd.DataFrame:
    row = {}
    for col in _feature_order:
        val = txn_dict.get(col)
        if val is None:
            if col in _freq_maps:
                val = 0.0
            elif col in _encoders:
                le = _encoders[col]
                val = le.transform([le.classes_[0]])[0]
            elif col in _feature_medians:
                val = _feature_medians[col]
            else:
                val = 0.0
        else:
            if col in _freq_maps:
                freq_map = _freq_maps[col]
                val = float(freq_map.get(val, 0.0))
            elif col in _encoders:
                le = _encoders[col]
                val = le.transform([str(val)])[0] if str(val) in le.classes_ else le.transform([le.classes_[0]])[0]
        row[col] = val
    return pd.DataFrame([row], columns=_feature_order)


def score_transaction_ieee(txn: IEEETransactionIn, drift: DriftResult | None = None) -> ScoreResult:
    raw = txn.model_dump()
    X = _encode_transaction(raw)
    proba = float(_model.predict_proba(X)[0, 1])

    shap_values = _explainer.shap_values(X)
    if isinstance(shap_values, list):
        contributions = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif shap_values.ndim == 3:
        contributions = shap_values[0, :, 1]
    else:
        contributions = shap_values[0]

    all_contributions = list(zip(_feature_order, contributions))
    positive = sorted(
        [(f, float(v)) for f, v in all_contributions if v > MIN_CONTRIBUTION],
        key=lambda x: -x[1],
    )
    named = [(f, v) for f, v in positive if _is_interpretable(f)][:TOP_N_EVIDENCE]
    other_sum = sum(v for f, v in positive if not _is_interpretable(f))

    evidence: list[Evidence] = []
    for feature, contribution in named:
        strength = round(min(1.0, contribution), 4)
        provided_value = raw.get(feature)
        if provided_value is None:
            value_str = "(not provided by caller; model used a training-set default)"
        else:
            value_str = str(provided_value)
        evidence.append(Evidence(
            signal=feature,
            direction=EvidenceDirection.SUPPORTS_FRAUD,
            strength=strength,
            description=f"{feature}: {value_str} (SHAP contribution +{contribution:.4f})",
        ))

    if other_sum > MIN_CONTRIBUTION:
        evidence.append(Evidence(
            signal="engineered_signals_combined",
            direction=EvidenceDirection.SUPPORTS_FRAUD,
            strength=round(min(1.0, other_sum), 4),
            description=f"Additional anonymized/engineered features contributed a combined "
                        f"+{other_sum:.4f} toward this score (not individually interpretable)",
        ))

    if drift is not None and drift.drift_score >= 0.3:
        evidence.append(Evidence(
            signal="pattern_evasion",
            direction=EvidenceDirection.SUPPORTS_FRAUD,
            strength=round(min(1.0, drift.drift_score * 0.1), 4),
            description=f"Agent 1 flagged pattern/evasion signals: {'; '.join(drift.drift_signals)}",
        ))

    return ScoreResult(
        transaction_id=txn.transaction_id,
        score=round(proba, 4),
        evidence=evidence,
        model_version="lightgbm-ieee-v3",
    )
