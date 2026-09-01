"""
Generates interpretable evidence from the V3 model's SHAP output. Model sees all 150
selected features (including anonymized V-columns for max predictive power), but only
named, human-legible features are shown as individual evidence -- V-columns get honestly
summarized as a combined "additional engineered signals" contribution, never hidden.

Run with: python scripts/generate_shap_evidence.py
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import shap

ARTIFACT_DIR = Path("app/ml_artifacts_ieee_v3")
TRANSACTION_PATH = "data/train_transaction.csv"
IDENTITY_PATH = "data/train_identity.csv"

# Feature-name prefixes/exact-names considered "interpretable enough to show individually"
INTERPRETABLE_PREFIXES = ("card", "TransactionAmt", "addr", "dist", "P_email", "R_email",
                           "D1", "D2", "D3", "D4", "D5", "D6", "D8", "D9", "D10", "D11",
                           "D12", "D13", "D14", "D15", "C1", "C2", "C3", "C4", "C5", "C6",
                           "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                           "hour_of_day", "day_of_week", "has_identity_data", "DeviceType",
                           "ProductCD", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9")

def is_interpretable(feature_name):
    return any(feature_name.startswith(p) for p in INTERPRETABLE_PREFIXES) and not feature_name.startswith("id_")

with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(ARTIFACT_DIR / "encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open(ARTIFACT_DIR / "freq_maps.pkl", "rb") as f:
    freq_maps = pickle.load(f)
with open(ARTIFACT_DIR / "top_features.pkl", "rb") as f:
    top_features = pickle.load(f)

interpretable_count = sum(1 for f in top_features if is_interpretable(f))
print(f"{interpretable_count} of {len(top_features)} selected features are interpretable")
print("Interpretable features:", [f for f in top_features if is_interpretable(f)][:20])

print("\nReloading test data (same pipeline as before)...")
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

# Grab 3 real fraud cases from the test set to demo evidence generation on
fraud_idx = y_test[y_test == 1].index[:3]
sample = X_test.loc[fraud_idx]

print(f"\nGenerating SHAP evidence for {len(sample)} real fraud test cases...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

results = []
for i, idx in enumerate(sample.index):
    contributions = list(zip(top_features, shap_values[i]))
    positive = sorted([(f, float(v)) for f, v in contributions if v > 0], key=lambda x: -x[1])
    named_evidence = [(f, v) for f, v in positive if is_interpretable(f)][:5]
    other_sum = sum(v for f, v in positive if not is_interpretable(f))

    print(f"\n--- Transaction {idx} (real fraud case) ---")
    for f, v in named_evidence:
        print(f"  {f}: {v:.4f}")
    if other_sum > 0:
        print(f"  [+ additional engineered signals, combined: {other_sum:.4f}]")

    results.append({
        "named_evidence": named_evidence,
        "other_engineered_signals_combined": round(other_sum, 4),
    })

with open(ARTIFACT_DIR / "sample_evidence_output.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved sample evidence output to {ARTIFACT_DIR}/sample_evidence_output.json")
