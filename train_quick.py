"""
Quick training script - no MLflow/matplotlib needed.
Trains 3 models on processed data and saves best_model.pkl + scaler.pkl
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading processed data...")
X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── Train models ───────────────────────────────────────────────────────────────
models = {
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    "RandomForest":       RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost":            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                        eval_metric="logloss", verbosity=0),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    results[name] = {"auc": auc, "f1": f1}
    print(f"  ROC AUC: {auc:.4f}  F1: {f1:.4f}")

# ── Pick best model ────────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["auc"])
best_model = models[best_name]
print(f"\n==> Best model: {best_name} (ROC AUC: {results[best_name]['auc']:.4f})")

# ── Save best_model.pkl ────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Saved models/best_model.pkl")

# ── Save a dummy scaler.pkl (data already scaled in preprocessing) ─────────────
# The API checks for scaler.pkl; save an identity scaler so the API won't skip it.
from sklearn.preprocessing import StandardScaler
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
scaler.fit(X_train[numeric_cols])
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved models/scaler.pkl")

print("\nDone! Restart uvicorn to load the new model.")
