"""
Finalizes the V3 model for serving: computes feature medians (for imputing missing fields
at inference time -- V3's training script never saved these), runs a cost-optimal threshold
sweep using the real Rs94/Rs34802 figures, and verifies SHAP evidence generation works on
this LightGBM model.

UNVERIFIED -- this was not test-run before being handed off. Please paste back the full
output, especially any errors, particularly around the SHAP section.

Run with: python scripts/finalize_ieee_v3.py
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

print("Loading saved V3 artifacts...")
with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(ARTIFACT_DIR / "encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open(ARTIFACT_DIR / "freq_maps.pkl", "rb") as f:
    freq_maps = pickle.load(f)
with open(ARTIFACT_DIR / "top_features.pkl", "rb") as f:
    top_features = pickle.load(f)

print(f"Loaded model, {len(top_features)} selected features")

# Rebuild the exact same data pipeline as V3's training script, so the test split matches
print("Reloading and re-cleaning full dataset (same steps as train_ieee_v3.py)...")
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
drop_cols = missing_pct[missing_pct > 0.85].index.tolist()
df = df.drop(columns=drop_cols)

missing_pct = df.isna().mean()
moderate_missing = [c for c in missing_pct[(missing_pct > 0.02) & (missing_pct <= 0.85)].index if c != "isFraud"]
missing_flags = pd.DataFrame({f"{c}_was_missing": df[c].isna().astype(int) for c in moderate_missing})
df = pd.concat([df, missing_flags], axis=1)
for col in moderate_missing:
    if df[col].dtype in ("float64", "int64"):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("unknown")
for col in df.columns[df.isna().any()].tolist():
    if df[col].dtype in ("float64", "int64"):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("unknown")

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

# NEW: save feature medians now, for use at inference time when a field is missing
feature_medians = {}
for col in top_features:
    if col in df_no_id.columns and df_no_id[col].dtype in ("float64", "int64", "float32", "int32"):
        feature_medians[col] = float(df_no_id[col].median())
with open(ARTIFACT_DIR / "feature_medians.pkl", "wb") as f:
    pickle.dump(feature_medians, f)
print(f"Saved {len(feature_medians)} feature medians for inference-time imputation")

X = df_no_id[top_features]
y = df_no_id["isFraud"]

split_idx = int(len(df_no_id) * 0.8)
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"Test set: {X_test.shape}, fraud count: {int(y_test.sum())}")

print("\nComputing probabilities...")
probs = model.predict_proba(X_test)[:, 1]

print("\n=== Probability distribution ===")
fraud_probs = probs[y_test.values == 1]
legit_probs = probs[y_test.values == 0]
print(f"Max probability: {probs.max():.4f}")
print(f"Fraud probs -- p10/p25/p50/p75/p90: {np.percentile(fraud_probs, [10,25,50,75,90]).round(4)}")
print(f"Legit probs -- p50/p75/p90/p95/p99: {np.percentile(legit_probs, [50,75,90,95,99]).round(4)}")

print("\n=== Cost-optimal threshold sweep (FP=Rs94, FN=Rs34802) ===")
best = None
for t in np.arange(0.01, 0.95, 0.01):
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
    cost = fp * COST_FP + fn * COST_FN
    if best is None or cost < best["cost"]:
        best = {"threshold": round(float(t), 3), "cost": int(cost), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

precision = best["tp"] / (best["tp"] + best["fp"]) if (best["tp"] + best["fp"]) else 0
recall = best["tp"] / (best["tp"] + best["fn"]) if (best["tp"] + best["fn"]) else 0
best["precision"] = round(precision, 4)
best["recall"] = round(recall, 4)
print(json.dumps(best, indent=2))

with open(ARTIFACT_DIR / "threshold_results.json", "w") as f:
    json.dump(best, f, indent=2)

print("\n=== SHAP check (this is the part I could not verify myself) ===")
try:
    import shap
    explainer = shap.TreeExplainer(model)
    sample = X_test.iloc[:3]
    shap_out = explainer.shap_values(sample)

    print(f"Type of shap_values output: {type(shap_out)}")
    if isinstance(shap_out, list):
        print(f"It's a list of length {len(shap_out)}")
        fraud_shap = shap_out[1] if len(shap_out) > 1 else shap_out[0]
    else:
        print(f"It's an array of shape {shap_out.shape}")
        if shap_out.ndim == 3:
            fraud_shap = shap_out[:, :, 1]
        else:
            fraud_shap = shap_out

    print(f"\nFinal fraud_shap shape: {fraud_shap.shape}")
    print("Top 5 SHAP contributors for first test transaction:")
    contributions = list(zip(top_features, fraud_shap[0]))
    contributions.sort(key=lambda x: -abs(x[1]))
    for feat, val in contributions[:5]:
        print(f"  {feat}: {val:.4f}")

    with open(ARTIFACT_DIR / "shap_output_shape_info.txt", "w") as f:
        f.write(f"Raw output type: {type(shap_out)}\n")
        f.write(f"Final fraud_shap shape used: {fraud_shap.shape}\n")

    print("\nSHAP check PASSED -- shapes handled correctly")
except Exception as e:
    print(f"\nSHAP check FAILED with error: {e}")
    print("Please paste this exact error back so it can be fixed.")

print("\nDone. Check app/ml_artifacts_ieee_v3/ for: feature_medians.pkl, threshold_results.json")
