"""
V2 training script: keeps the anonymized V-columns (real predictive signal, just
uninterpretable), uses frequency encoding on high-cardinality ID-like columns instead
of top-N label encoding, and retunes hyperparameters on the REAL base rate (3.5%),
using a proper time-based split.

The model trains on everything (including V-columns). Evidence/SHAP display for the
Reviewer Agent should later be filtered to only the named, interpretable features --
that's a separate step, this script is just about maximizing real model quality.

Run with: python scripts/train_ieee_v2.py
"""
import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier

TRANSACTION_PATH = "data/train_transaction.csv"
IDENTITY_PATH = "data/train_identity.csv"
OUT_DIR = Path("app/ml_artifacts_ieee_v2")
OUT_DIR.mkdir(exist_ok=True, parents=True)

FREQ_ENCODE_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2",
                     "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_31", "id_33"]

print("Loading FULL dataset WITH V-columns (this will take a while, it's a big file)...")
txn = pd.read_csv(TRANSACTION_PATH)
identity = pd.read_csv(IDENTITY_PATH)
print(f"Transactions: {txn.shape}, fraud rate: {txn['isFraud'].mean():.4f}")

df = txn.merge(identity, on="TransactionID", how="left")
df["has_identity_data"] = df["DeviceType"].notna()

SECONDS_PER_DAY = 60 * 60 * 24
df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
df["day_of_week"] = (df["TransactionDT"] // SECONDS_PER_DAY) % 7

missing_pct = df.isna().mean()
drop_cols = missing_pct[missing_pct > 0.85].index.tolist()
df = df.drop(columns=drop_cols)
print(f"Dropped {len(drop_cols)} columns with >85% missingness")

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

freq_maps = {}
for col in FREQ_ENCODE_COLS:
    if col not in df_no_id.columns:
        continue
    freq_map = df_no_id[col].value_counts(normalize=True)
    freq_maps[col] = freq_map
    df_no_id[col] = df_no_id[col].map(freq_map).fillna(0)

categorical_cols = [c for c in df_no_id.select_dtypes(include="object").columns if c not in FREQ_ENCODE_COLS]
bool_cols = df_no_id.select_dtypes(include="bool").columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_no_id[col] = le.fit_transform(df_no_id[col].astype(str))
    encoders[col] = le
for col in bool_cols:
    df_no_id[col] = df_no_id[col].astype(int)

feature_cols = [c for c in df_no_id.columns if c not in ("isFraud", "TransactionDT")]
X = df_no_id[feature_cols]
y = df_no_id["isFraud"]

split_idx = int(len(df_no_id) * 0.8)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
print(f"Train: {X_train.shape} (fraud={int(y_train.sum())}), Test: {X_test.shape} (fraud={int(y_test.sum())})")

spw = (y_train == 0).sum() / (y_train == 1).sum()

param_dist = {
    "n_estimators": [300, 500, 700],
    "max_depth": [4, 5, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 63],
    "min_child_samples": [10, 20],
}

print("Running hyperparameter search on REAL base rate (this will take longer than before)...")
t0 = time.time()
lgbm = LGBMClassifier(scale_pos_weight=spw, random_state=42, verbose=-1, n_jobs=-1)
search = RandomizedSearchCV(lgbm, param_dist, n_iter=12, scoring="average_precision", cv=3, random_state=42, n_jobs=1)
search.fit(X_train, y_train)
print(f"Search took {time.time()-t0:.1f}s")

best = search.best_estimator_
probs = best.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)
base_rate = y_test.mean()

print()
print("=" * 60)
print(f"Best params: {search.best_params_}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Base rate in test set: {base_rate:.4f}")
print(f"Lift over base rate: {pr_auc/base_rate:.1f}x")
print()
print("Comparison:")
print(f"  V1 full data (no V-cols, real base rate):    PR-AUC 0.5180, lift 14.8x")
print(f"  V2 (WITH V-cols, freq encoding, real rate):  PR-AUC {pr_auc:.4f}, lift {pr_auc/base_rate:.1f}x")
print(f"  Kavach:                                       PR-AUC 0.895, lift 71.6x")
print("=" * 60)

with open(OUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(best, f)
with open(OUT_DIR / "encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open(OUT_DIR / "freq_maps.pkl", "wb") as f:
    pickle.dump(freq_maps, f)
with open(OUT_DIR / "feature_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print(f"Saved to {OUT_DIR}/")
