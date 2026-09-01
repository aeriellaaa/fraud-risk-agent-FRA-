"""
Tests whether training on the FULL train_transaction.csv (not the trimmed sample)
improves PR-AUC, using the same time-based split methodology as before.
Run with: python scripts/train_full_data_test.py
NOTE: expects the FULL, untrimmed train_transaction.csv in data/ -- this will use
more memory and take longer than the sample-based scripts.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier

TRANSACTION_PATH = "data/train_transaction.csv"
IDENTITY_PATH = "data/train_identity.csv"
HIGH_CARDINALITY_TOP_N = 20
BEST_PARAMS = {"num_leaves": 31, "n_estimators": 700, "min_child_samples": 20, "max_depth": 6, "learning_rate": 0.05}

print("Loading FULL dataset (this will take longer and use more memory than before)...")
txn = pd.read_csv(TRANSACTION_PATH)
identity = pd.read_csv(IDENTITY_PATH)
v_cols = [c for c in txn.columns if c.startswith("V")]
txn = txn.drop(columns=v_cols)
print(f"Full transaction data: {txn.shape}, fraud rate: {txn['isFraud'].mean():.4f}")

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

df = df.drop(columns=["TransactionID"])
categorical_cols = df.select_dtypes(include="object").columns.tolist()
bool_cols = df.select_dtypes(include="bool").columns.tolist()
for col in categorical_cols:
    if df[col].nunique() > HIGH_CARDINALITY_TOP_N:
        top_values = df[col].value_counts().nlargest(HIGH_CARDINALITY_TOP_N).index
        df[col] = df[col].where(df[col].isin(top_values), other="other")
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
for col in bool_cols:
    df[col] = df[col].astype(int)

feature_cols = [c for c in df.columns if c not in ("isFraud", "TransactionDT")]

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]
X_train, y_train = train_df[feature_cols], train_df["isFraud"]
X_test, y_test = test_df[feature_cols], test_df["isFraud"]

print(f"Train: {X_train.shape} (fraud={int(y_train.sum())}), Test: {X_test.shape} (fraud={int(y_test.sum())})")

spw = (y_train == 0).sum() / (y_train == 1).sum()
print("Training on full data (this may take a few minutes)...")
model = LGBMClassifier(scale_pos_weight=spw, random_state=42, verbose=-1, n_jobs=-1, **BEST_PARAMS)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

pr_auc = average_precision_score(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)
print()
print(f"FULL DATA, time-based split: PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")
print(f"Compare to sample-based time-split result: PR-AUC=0.8363  ROC-AUC=0.9054")
