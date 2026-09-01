"""
Computes recall at fixed false-positive budgets (1%, 2%, 5% of test set), matching
kavach's own methodology, instead of an unconstrained cost-optimal threshold (which
produced an undeployable 71%-of-transactions-flagged result).

Run with: python scripts/threshold_budget_analysis.py
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix

ARTIFACT_DIR = Path("app/ml_artifacts_ieee_v3")
TRANSACTION_PATH = "data/train_transaction.csv"
IDENTITY_PATH = "data/train_identity.csv"
COST_FP = 94
COST_FN = 34802

with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(ARTIFACT_DIR / "encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open(ARTIFACT_DIR / "freq_maps.pkl", "rb") as f:
    freq_maps = pickle.load(f)
with open(ARTIFACT_DIR / "top_features.pkl", "rb") as f:
    top_features = pickle.load(f)

print("Reloading and re-cleaning full dataset...")
txn = pd.read_csv(TRANSACTION_PATH)
identity = pd.read_csv(IDENTITY_PATH)
df = txn.merge(identity, on="TransactionID", how="left")
new_cols = pd.DataFrame({
    "has_identity_data": df["DeviceType"].notna(),
    "hour_of_day": (df["TransactionDT"] // 3600) % 24,
    "day_of_week": (df["TransactionDT"] // 86400) % 7,
})
df = pd.concat([df, new_cols], axis=1)
missing_pct = df.isna().mean()
df = df.drop(columns=missing_pct[missing_pct > 0.85].index.tolist())
missing_pct = df.isna().mean()
moderate_missing = [c for c in missing_pct[(missing_pct > 0.02) & (missing_pct <= 0.85)].index if c != "isFraud"]
missing_flags = pd.DataFrame({f"{c}_was_missing": df[c].isna().astype(int) for c in moderate_missing})
df = pd.concat([df, missing_flags], axis=1)
for col in moderate_missing:
    df[col] = df[col].fillna(df[col].median()) if df[col].dtype in ("float64","int64") else df[col].fillna("unknown")
for col in df.columns[df.isna().any()].tolist():
    df[col] = df[col].fillna(df[col].median()) if df[col].dtype in ("float64","int64") else df[col].fillna("unknown")
df = df.sort_values("TransactionDT").reset_index(drop=True)
df_no_id = df.drop(columns=["TransactionID"])
for col, freq_map in freq_maps.items():
    if col in df_no_id.columns:
        df_no_id[col] = df_no_id[col].map(freq_map).fillna(0)
for col, le in encoders.items():
    if col in df_no_id.columns:
        known = set(le.classes_)
        df_no_id[col] = df_no_id[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
        df_no_id[col] = le.transform(df_no_id[col])

X = df_no_id[top_features]
y = df_no_id["isFraud"]
split_idx = int(len(df_no_id) * 0.8)
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
probs = model.predict_proba(X_test)[:, 1]
n_total = len(y_test)

def recall_at_fp_budget(y_test, probs, fp_budget_pct):
    max_fp = int(n_total * fp_budget_pct)
    best = None
    for t in np.unique(probs)[::-1]:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
        if fp <= max_fp:
            recall = tp / (tp+fn) if (tp+fn) else 0
            if best is None or recall > best["recall"]:
                best = {"threshold": round(float(t),4), "fp": int(fp), "fn": int(fn),
                        "tp": int(tp), "tn": int(tn), "recall": round(float(recall),4),
                        "cost_inr": int(fp*COST_FP + fn*COST_FN)}
        else:
            break
    return best

print(f"\nTest set size: {n_total}, real fraud count: {int(y_test.sum())}")
print("\n=== Recall at fixed FP budgets (operationally realistic) ===")
results = {}
for budget in [0.01, 0.02, 0.05, 0.10]:
    r = recall_at_fp_budget(y_test, probs, budget)
    results[f"{int(budget*100)}pct_fp_budget"] = r
    print(f"At {budget*100:.0f}% FP budget (max {int(n_total*budget)} reviews): {r}")

with open(ARTIFACT_DIR / "threshold_budget_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {ARTIFACT_DIR}/threshold_budget_results.json")
