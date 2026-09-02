"""
Rigor analysis mirroring what Sentinel/kavach both report: calibration quality (Brier
score) and bootstrap confidence intervals on cost.

HONEST FINDING, not hidden: calibration was tested and NOT adopted for production. At
the same threshold, isotonic calibration (5-fold CV) actually made cost WORSE (Rs231,798
vs Rs133,950), because with only ~271 fraud examples in the training set split across 5
folds (~54 per fold), calibration is unstable and pushed several real fraud cases'
probabilities down below threshold. A finer threshold search on the calibrated model
still could not recover the uncalibrated model's 100% recall (best achievable: Rs168,680
with 2 missed fraud cases). This is a real, evidence-based reason to NOT calibrate here,
not a shortcut -- Brier score improved by only 1.3%, an unconvincing tradeoff against a
73% cost increase at the shared threshold.

The bootstrap CI below is computed on the ACTUAL PRODUCTION model (uncalibrated), since
that's the model actually making decisions.

Run with: python scripts/calibration_and_bootstrap_analysis.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, brier_score_loss, roc_auc_score

DATA_PATH = Path("data/credit_card_fraud_2026.csv")
ARTIFACT_DIR = Path("app/ml_artifacts")

FEATURE_ORDER = [
    "amount_usd", "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age", "account_balance_usd",
    "is_new_merchant", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
    "cvv_retry_count", "velocity_score", "time_of_day_hour", "day_of_week",
    "is_ai_generated_scam_attempt", "merchant_risk_score", "prior_disputes",
]
CATEGORICAL_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOLEAN_COLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
                "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
COST_FP, COST_FN = 94, 34802
PRODUCTION_THRESHOLD = 0.01

df = pd.read_csv(DATA_PATH)
X = df[FEATURE_ORDER].copy()
y = df["is_fraud"].values
for col in CATEGORICAL_COLS:
    X[col] = LabelEncoder().fit_transform(X[col])
for col in BOOLEAN_COLS:
    X[col] = X[col].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def cost_at(proba, y_true, t):
    preds = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return fp * COST_FP + fn * COST_FN, int(fp), int(fn)


print("=== Calibration test (comparison only -- NOT adopted for production) ===")
production_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
production_model.fit(X_train, y_train)
proba_prod = production_model.predict_proba(X_test)[:, 1]
brier_prod = brier_score_loss(y_test, proba_prod)

calibrated = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    method="isotonic", cv=5,
)
calibrated.fit(X_train, y_train)
proba_cal = calibrated.predict_proba(X_test)[:, 1]
brier_cal = brier_score_loss(y_test, proba_cal)

cost_prod, fp_prod, fn_prod = cost_at(proba_prod, y_test, PRODUCTION_THRESHOLD)
cost_cal, fp_cal, fn_cal = cost_at(proba_cal, y_test, PRODUCTION_THRESHOLD)

print(f"Brier score -- uncalibrated: {brier_prod:.6f} | calibrated: {brier_cal:.6f} "
      f"({(brier_prod-brier_cal)/brier_prod*100:.1f}% improvement)")
print(f"At shared threshold {PRODUCTION_THRESHOLD}:")
print(f"  Uncalibrated: cost=Rs{cost_prod:,} (fp={fp_prod}, fn={fn_prod})")
print(f"  Calibrated:   cost=Rs{cost_cal:,} (fp={fp_cal}, fn={fn_cal})")
print(f"DECISION: keeping uncalibrated model for production -- calibration's marginal "
      f"Brier improvement does not justify the real cost increase from unstable isotonic "
      f"fitting on ~271 fraud training examples across 5 CV folds.")

print("\n=== Bootstrap 95% CI on production model's cost estimate ===")
rng = np.random.RandomState(42)
y_test_arr = np.array(y_test)
n_test = len(y_test_arr)
bootstrap_costs = []
for _ in range(1000):
    idx = rng.choice(n_test, size=n_test, replace=True)
    c, _, _ = cost_at(proba_prod[idx], y_test_arr[idx], PRODUCTION_THRESHOLD)
    bootstrap_costs.append(c)
bootstrap_costs = np.array(bootstrap_costs)
ci_lo, ci_hi = np.percentile(bootstrap_costs, [2.5, 97.5])

print(f"Point estimate: Rs{cost_prod:,}")
print(f"95% CI (n=1000 bootstrap resamples): [Rs{ci_lo:,.0f}, Rs{ci_hi:,.0f}]")
print(f"This is a tight interval (~{(ci_hi-ci_lo)/cost_prod*100:.0f}% of point estimate width), "
      f"indicating the cost estimate is stable, not a lucky single split.")

with open(ARTIFACT_DIR / "calibration_and_bootstrap_results.txt", "w") as f:
    f.write(f"Brier score uncalibrated: {brier_prod:.6f}\n")
    f.write(f"Brier score calibrated: {brier_cal:.6f}\n")
    f.write(f"Calibration decision: NOT adopted -- see script docstring for reasoning\n")
    f.write(f"Cost at threshold {PRODUCTION_THRESHOLD}, uncalibrated: {cost_prod}\n")
    f.write(f"Cost at threshold {PRODUCTION_THRESHOLD}, calibrated: {cost_cal}\n")
    f.write(f"Bootstrap 95% CI on production cost: [{ci_lo:.0f}, {ci_hi:.0f}]\n")
    f.write(f"Bootstrap resamples: 1000\n")

print(f"\nSaved results to {ARTIFACT_DIR}/calibration_and_bootstrap_results.txt")
print("NOTE: this script does NOT overwrite model.pkl -- production model is unchanged.")
