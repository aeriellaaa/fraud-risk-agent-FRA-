"""
V3: fixes two real mistakes from V2 --
1. Adds back subsample/colsample_bytree to the search grid (V2 accidentally dropped these).
2. Does feature selection first: quick fit to rank importance, keeps only the top 150
   features instead of blindly using all ~340 V-columns.

Run with: python scripts/train_ieee_v3.py
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
OUT_DIR = Path("app/ml_artifacts_ieee_v3")
OUT_DIR.mkdir(exist_ok=True, parents=True)
TOP_N_FEATURES = 150

FREQ_ENCODE_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2",
                     "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_31", "id_33"]

print("Loading FULL dataset WITH V-columns...")
txn = pd.read_csv(TRANSACTION_PATH)
identity = pd.read_csv(IDENTITY_PATH)
print(f"Transactions: {txn.shape}, fraud rate: {txn['isFraud'].mean():.4f}")

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

all_feature_cols = [c for c in df_no_id.columns if c not in ("isFraud", "TransactionDT")]
X_all = df_no_id[all_feature_cols]
y = df_no_id["isFraud"]

split_idx = int(len(df_no_id) * 0.8)
X_train_all, y_train = X_all.iloc[:split_idx], y.iloc[:split_idx]
X_test_all, y_test = X_all.iloc[split_idx:], y.iloc[split_idx:]

print(f"\nStep 1: quick fit on all {len(all_feature_cols)} features to rank importance...")
t0 = time.time()
spw = (y_train == 0).sum() / (y_train == 1).sum()
quick_model = LGBMClassifier(n_estimators=200, max_depth=6, scale_pos_weight=spw, random_state=42, verbose=-1, n_jobs=-1)
quick_model.fit(X_train_all, y_train)
print(f"Quick fit took {time.time()-t0:.1f}s")

importances = pd.Series(quick_model.feature_importances_, index=all_feature_cols).sort_values(ascending=False)
top_features = importances.head(TOP_N_FEATURES).index.tolist()
print(f"Selected top {TOP_N_FEATURES} features. Top 15:")
print(importances.head(15))

X_train = X_train_all[top_features]
X_test = X_test_all[top_features]

param_dist = {
    "n_estimators": [300, 500, 700],
    "max_depth": [4, 5, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 63],
    "min_child_samples": [10, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.85],
}

print(f"\nStep 2: hyperparameter search on top {TOP_N_FEATURES} features (with subsample/colsample this time)...")
t0 = time.time()
lgbm = LGBMClassifier(scale_pos_weight=spw, random_state=42, verbose=-1, n_jobs=-1)
search = RandomizedSearchCV(lgbm, param_dist, n_iter=15, scoring="average_precision", cv=3, random_state=42, n_jobs=1)
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
print(f"Lift over base rate: {pr_auc/base_rate:.1f}x")
print()
print("Comparison:")
print(f"  V1 (no V-cols, all features):                PR-AUC 0.5180, lift 14.8x")
print(f"  V2 (all 608 features, no col-subsampling):   PR-AUC 0.5034, lift 14.6x")
print(f"  V3 (top {TOP_N_FEATURES} features, proper regularization): PR-AUC {pr_auc:.4f}, lift {pr_auc/base_rate:.1f}x")
print(f"  Kavach:                                        PR-AUC 0.895, lift 71.6x")
print("=" * 60)

with open(OUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(best, f)
with open(OUT_DIR / "encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open(OUT_DIR / "freq_maps.pkl", "wb") as f:
    pickle.dump(freq_maps, f)
with open(OUT_DIR / "top_features.pkl", "wb") as f:
    pickle.dump(top_features, f)
print(f"Saved to {OUT_DIR}/")
