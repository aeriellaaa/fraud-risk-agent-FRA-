"""
Real precision/recall/cost metrics from the trained model, loaded from
scripts/train_model.py's output, plus a baseline comparison from
scripts/baseline_comparison.py and a real cost curve from
scripts/generate_cost_curve.py.
"""

from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()

RESULTS_FILE = Path("app/ml_artifacts/training_results.txt")
BASELINE_FILE = Path("app/ml_artifacts/baseline_comparison.txt")
COST_CURVE_FILE = Path("app/ml_artifacts/cost_curve.json")


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


def _parse_baseline_structured() -> list[dict] | None:
    """Parses baseline_comparison.txt's fixed-width table into structured rows."""
    if not BASELINE_FILE.exists():
        return None
    lines = BASELINE_FILE.read_text().strip().split("\n")
    rows = []
    for line in lines[1:]:  # skip header
        if not line.strip() or line.strip().startswith("Base rate"):
            continue
        parts = line.split()
        # last 5 tokens are ROC-AUC, PR-AUC, Threshold, Precision, Recall, Cost -- label is everything before
        try:
            cost = int(parts[-1].replace(",", ""))
            recall = float(parts[-2])
            precision = float(parts[-3])
            threshold = float(parts[-4])
            pr_auc = float(parts[-5])
            roc_auc = float(parts[-6])
            label = " ".join(parts[:-6])
            rows.append({
                "label": label,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "cost_inr": cost,
            })
        except (ValueError, IndexError):
            continue
    return rows or None


def _read_cost_curve() -> dict | None:
    if not COST_CURVE_FILE.exists():
        return None
    return json.loads(COST_CURVE_FILE.read_text())


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

    baseline_comparison: structured list (Rules engine, Naive, Random Forest),
    parsed from baseline_comparison.txt. cost_curve: real threshold sweep from
    generate_cost_curve.py. Both null if their generating script hasn't been run.
    """
    parsed = _parse_results()
    if "error" in parsed:
        return parsed

    baseline_structured = _parse_baseline_structured()
    cost_curve = _read_cost_curve()

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
        "baseline_comparison_raw": _read_baseline(),
        "baseline_comparison": baseline_structured,  # null if not yet generated
        "cost_curve": cost_curve,  # null if scripts/generate_cost_curve.py hasn't been run
    }


def _read_baseline() -> str:
    if not BASELINE_FILE.exists():
        return "Not yet generated. Run scripts/baseline_comparison.py first."
    return BASELINE_FILE.read_text()
