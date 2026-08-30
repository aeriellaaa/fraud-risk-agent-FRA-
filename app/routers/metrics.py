"""
Real precision/recall/cost metrics from the trained model, loaded from
scripts/train_model.py's output.
"""

from fastapi import APIRouter
from pathlib import Path

router = APIRouter()

RESULTS_FILE = Path("app/ml_artifacts/training_results.txt")


def _parse_results() -> dict:
    if not RESULTS_FILE.exists():
        return {"error": "Model not yet trained. Run scripts/train_model.py first."}

    text = RESULTS_FILE.read_text()
    result = {}
    for line in text.strip().split("\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


@router.get("/metrics")
def get_metrics():
    """
    Real, measured precision/recall/AUC and cost-optimal threshold from
    a held-out test set (20% of credit_card_fraud_2026.csv, stratified
    split, random_state=42 -- see scripts/train_model.py).

    Cost figures: FP=Rs94 (avg. Indian fraud analyst hourly rate, 15 min
    review, ERI SalaryExpert) and FN=Rs34802 (avg. value of card/internet
    banking fraud in India, FY22, Lok Sabha data via Business Standard).
    Both sourced -- see scripts/train_model.py header for full citations.
    """
    parsed = _parse_results()
    if "error" in parsed:
        return parsed

    return {
        "model": "RandomForestClassifier (n_estimators=300, class_weight=balanced)",
        "held_out_test_set_size": 4000,
        "auc": float(parsed.get("AUC", 0)),
        "precision": float(parsed.get("Precision", 0)),
        "recall": float(parsed.get("Recall", 0)),
        "cost_optimal_threshold": float(parsed.get("Best threshold", 0)),
        "cost_optimal_total_cost_inr": float(parsed.get("Best cost", 0)),
        "default_threshold_cost_inr": float(parsed.get("Default 0.5 threshold cost", 0)),
        "confusion_matrix_at_optimal_threshold": parsed.get("Confusion matrix", "unavailable"),
        "cost_figures_source": "FP=Rs94 (ERI SalaryExpert), FN=Rs34802 (Lok Sabha data via "
                                "Business Standard, FY22 card/internet fraud). Both sourced, "
                                "see scripts/train_model.py header.",
    }
