"""
Cleans and merges the IEEE-CIS sample data into a model-ready dataset.
Run with: python scripts/prepare_ieee_data.py
"""
import pandas as pd
import numpy as np

TRANSACTION_PATH = "data/train_transaction_sample.csv"
IDENTITY_PATH = "data/train_identity.csv"
OUTPUT_PATH = "data/ieee_cis_cleaned.csv"

print("Loading data...")
txn = pd.read_csv(TRANSACTION_PATH)
identity = pd.read_csv(IDENTITY_PATH)

print(f"Transactions: {txn.shape}, Identity: {identity.shape}")

df = txn.merge(identity, on="TransactionID", how="left")
df["has_identity_data"] = df["DeviceType"].notna()
print(f"Merged: {df.shape}, rows with identity data: {df['has_identity_data'].sum()}")

SECONDS_PER_DAY = 60 * 60 * 24
df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
df["day_of_week"] = (df["TransactionDT"] // SECONDS_PER_DAY) % 7

missing_pct = df.isna().mean()
drop_cols = missing_pct[missing_pct > 0.85].index.tolist()
df = df.drop(columns=drop_cols)
print(f"Dropped {len(drop_cols)} columns with >85% missingness: {drop_cols}")

missing_pct = df.isna().mean()
moderate_missing = missing_pct[(missing_pct > 0.02) & (missing_pct <= 0.85)].index.tolist()
moderate_missing = [c for c in moderate_missing if c not in ("isFraud",)]

missing_flags = pd.DataFrame({f"{col}_was_missing": df[col].isna().astype(int) for col in moderate_missing})
df = pd.concat([df, missing_flags], axis=1)
for col in moderate_missing:
    if df[col].dtype in ("float64", "int64"):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("unknown")

print(f"Added was_missing flags + imputed {len(moderate_missing)} columns")

remaining_null_cols = df.columns[df.isna().any()].tolist()
for col in remaining_null_cols:
    if df[col].dtype in ("float64", "int64"):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("unknown")

print(f"Final shape: {df.shape}")
print(f"Any nulls left: {df.isna().any().any()}")
print(f"Fraud rate: {df['isFraud'].mean():.4f}")

df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
