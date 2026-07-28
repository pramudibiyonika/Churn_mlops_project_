"""
Part 2 - Model Evaluation
Evaluates best model from MLflow and saves results to reports/
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (accuracy_score, precision_score,
                            recall_score, f1_score, roc_auc_score,
                            confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import pickle
import json
import os
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def load_test_data():
    """Load test data"""
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    print(f"Test data loaded: {X_test.shape}")
    return X_test, y_test


def get_best_run_from_mlflow():
    """Get best model run from MLflow by ROC AUC"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("churn-prediction")

    if experiment is None:
        raise ValueError("MLflow experiment 'churn-prediction' not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"]
    )

    if not runs:
        raise ValueError("No runs found. Run train.py first.")

    best_run = runs[0]
    best_auc = best_run.data.metrics.get("roc_auc", 0)

    print(f"Best Model: {best_run.info.run_name}")
    print(f"Best ROC AUC: {best_auc:.4f}")
    return best_run


def load_best_model():
    """Load best model from models/best_model.pkl"""
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded: {type(model).__name__}")
    return model


def save_evaluation_plots(y_test, y_pred, y_prob, model_name):
    """Save confusion matrix and ROC curve"""
    os.makedirs("reports", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.colorbar(im)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    plt.savefig("reports/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig("reports/roc_curve.png")
    plt.close()

    print("Plots saved to reports/")


def run_evaluation():
    """Main evaluation function - called by Airflow DAG"""
    # Load test data
    X_test, y_test = load_test_data()

    # Load best model
    model = load_best_model()
    model_name = type(model).__name__

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
    }

    # Print results
    print("\n========== Evaluation Results ==========")
    print(f"Model:     {model_name}")
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1 Score:  {metrics['f1_score']}")
    print(f"ROC AUC:   {metrics['roc_auc']}")
    print("=========================================")

    # Save metrics to reports/evaluation.json
    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to reports/evaluation.json")

    # Save plots
    save_evaluation_plots(y_test, y_pred, y_prob, model_name)

    # Save best model info
    with open("models/best_model_info.txt", "w") as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']}\n")
        f.write(f"F1 Score: {metrics['f1_score']}\n")

    print("Evaluation complete!")
    return metrics


if __name__ == "__main__":
    run_evaluation()