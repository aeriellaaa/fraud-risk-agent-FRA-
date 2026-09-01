import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier

DATA_PATH = "data/ieee_cis_cleaned.csv"
HIGH_CARDINALITY_TOP_N = 20

df = pd.read_csv(DATA_PATH)
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

feature_cols = [c for c in df.columns if c != "isFraud"]
X = df[feature_cols]
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
spw = (y_train == 0).sum() / (y_train == 1).sum()

param_dist = {
    "n_estimators": [300, 500, 700],
    "max_depth": [4, 5, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 63],
    "min_child_samples": [10, 20],
}

t0 = time.time()
lgbm = LGBMClassifier(scale_pos_weight=spw, random_state=42, verbose=-1, n_jobs=-1)
search = RandomizedSearchCV(lgbm, param_dist, n_iter=12, scoring="average_precision", cv=StratifiedKFold(3), random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
probs = best.predict_proba(X_test)[:, 1]
print(f"Search took {time.time()-t0:.1f}s")
print(f"Best params: {search.best_params_}")
print(f"Tuned LightGBM: PR-AUC={average_precision_score(y_test, probs):.4f}  ROC-AUC={roc_auc_score(y_test, probs):.4f}")
