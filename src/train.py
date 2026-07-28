"""
Part 2 - Model Development
Train 3 models with MLflow logging, confusion matrix, ROC curve.
Saves best model to models/best_model.pkl
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                            recall_score, f1_score, roc_auc_score,
                            confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import os
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to DAGsHub MLflow (token used as both username and password)
DAGSHUB_USER  = "thilanisenarath403"
DAGSHUB_TOKEN = "50d70aa9dfb5d3d193d2290a27001d855dd8650a"
DAGSHUB_URI   = f"https://dagshub.com/{DAGSHUB_USER}/churn-mlops-project.mlflow"

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_TOKEN
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(DAGSHUB_URI)
print(f"MLflow → DAGsHub: {DAGSHUB_URI}")

mlflow.set_experiment("churn-prediction")

def load_data():
    """Load processed data"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics, y_pred, y_prob

def save_confusion_matrix(y_test, y_pred, model_name):
    """Save confusion matrix as image"""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.colorbar(im)

    # Add numbers inside boxes
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    path = f"models/{model_name}_confusion_matrix.png"
    os.makedirs("models", exist_ok=True)
    plt.savefig(path)
    plt.close()
    return path

def save_roc_curve(y_test, y_prob, model_name):
    """Save ROC curve as image"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    path = f"models/{model_name}_roc_curve.png"
    plt.savefig(path)
    plt.close()
    return path

def train_model(model, model_name, params, X_train, X_test, y_train, y_test):
    """Train model and log everything to MLflow"""
    print(f"\nTraining {model_name}...")

    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Save and log confusion matrix
        cm_path = save_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)

        # Save and log ROC curve
        roc_path = save_roc_curve(y_test, y_prob, model_name)
        mlflow.log_artifact(roc_path)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        # Print results
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    return metrics

def save_best_model(all_metrics, all_models):
    """Compare models and save the best one"""
    best_model_name = max(all_metrics, key=lambda x: all_metrics[x]["roc_auc"])
    best_model = all_models[best_model_name]
    best_metrics = all_metrics[best_model_name]

    # Save best model
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save best metrics to reports
    import json
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            **best_metrics
        }, f, indent=4)

    print(f"\n========== Model Comparison ==========")
    for name, metrics in all_metrics.items():
        print(f"{name:25s} ROC AUC: {metrics['roc_auc']:.4f}  F1: {metrics['f1_score']:.4f}")
    print(f"======================================")
    print(f"Best Model: {best_model_name} (ROC AUC: {best_metrics['roc_auc']:.4f})")
    print(f"Best model saved to models/best_model.pkl")
    print(f"Metrics saved to reports/metrics.json")

    return best_model_name, best_model


def run_training():
    """Main training function - called by Airflow DAG"""
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    print("Data loaded successfully!")

    # Model 1 - Logistic Regression
    lr_params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    lr_model = LogisticRegression(**lr_params)
    lr_metrics = train_model(
        lr_model, "LogisticRegression",
        lr_params, X_train, X_test, y_train, y_test
    )

    # Model 2 - Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    rf_model = RandomForestClassifier(**rf_params)
    rf_metrics = train_model(
        rf_model, "RandomForest",
        rf_params, X_train, X_test, y_train, y_test
    )

    # Model 3 - XGBoost
    xgb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    xgb_model = XGBClassifier(**xgb_params, eval_metric="logloss")
    xgb_metrics = train_model(
        xgb_model, "XGBoost",
        xgb_params, X_train, X_test, y_train, y_test
    )

    # Save best model
    all_metrics = {
        "LogisticRegression": lr_metrics,
        "RandomForest": rf_metrics,
        "XGBoost": xgb_metrics
    }
    all_models = {
        "LogisticRegression": lr_model,
        "RandomForest": rf_model,
        "XGBoost": xgb_model
    }

    best_model_name, best_model = save_best_model(all_metrics, all_models)
    print("\nAll models trained and logged to MLflow!")
    return best_model_name


if __name__ == "__main__":
    run_training()