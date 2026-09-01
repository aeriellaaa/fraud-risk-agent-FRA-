"""
Trains the final LightGBM model on IEEE-CIS data using the best params found by the
search, saves the model + encoders, and checks the probability distribution.
Run with: python scripts/train_ieee_final.py
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier

DATA_PATH = "data/ieee_cis_cleaned.csv"
HIGH_CARDINALITY_TOP_N = 20
OUT_DIR = Path("app/ml_artifacts_ieee")
OUT_DIR.mkdir(exist_ok=True)

BEST_PARAMS = {
    "num_leaves": 31, "n_estimators": 700, "min_child_samples": 20,
    "max_depth": 6, "learning_rate": 0.05,
}

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["TransactionID"])
categorical_cols = df.select_dtypes(include="object").columns.tolist()
bool_cols = df.select_dtypes(include="bool").columns.tolist()

encoders = {}
top_value_lists = {}
for col in categorical_cols:
    if df[col].nunique() > HIGH_CARDINALITY_TOP_N:
        top_values = df[col].value_counts().nlargest(HIGH_CARDINALITY_TOP_N).index.tolist()
        top_value_lists[col] = top_values
        df[col] = df[col].where(df[col].isin(top_values), other="other")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
for col in bool_cols:
    df[col] = df[col].astype(int)

feature_cols = [c for c in df.columns if c != "isFraud"]
X = df[feature_cols]
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
spw = (y_train == 0).sum() / (y_train == 1).sum()

print("Training final model with best params...")
model = LGBMClassifier(scale_pos_weight=spw, random_state=42, verbose=-1, n_jobs=-1, **BEST_PARAMS)
model.fit(X_train, y_train)

probs_test = model.predict_proba(X_test)[:, 1]
print(f"PR-AUC: {average_precision_score(y_test, probs_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probs_test):.4f}")

print()
print("=== Probability distribution check ===")
fraud_probs = probs_test[y_test.values == 1]
legit_probs = probs_test[y_test.values == 0]
print(f"Max probability overall: {probs_test.max():.4f}")
print(f"Fraud probs -- p10/p25/p50/p75/p90: {np.percentile(fraud_probs, [10,25,50,75,90]).round(4)}")
print(f"Legit probs -- p50/p75/p90/p95/p99: {np.percentile(legit_probs, [50,75,90,95,99]).round(4)}")

with open(OUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(OUT_DIR / "encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open(OUT_DIR / "top_value_lists.pkl", "wb") as f:
    pickle.dump(top_value_lists, f)
with open(OUT_DIR / "feature_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print()
print(f"Saved model, encoders, top_value_lists, feature_columns to {OUT_DIR}/")
