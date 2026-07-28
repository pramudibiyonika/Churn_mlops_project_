# Airflow DAG - Orchestration developed by: pramudibiyonika (Cit-23-02-0345)
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

PROJECT_DIR = "/mnt/c/Users/Admin/Desktop/churn-mlops-project"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='Churn Prediction MLOps Pipeline',
    schedule='@daily',
    catchup=False
)

def run_data_ingestion():
    import pandas as pd
    df = pd.read_csv(f"{PROJECT_DIR}/data/raw/telco_customer_churn_data.csv")
    os.makedirs(f"{PROJECT_DIR}/data/processed", exist_ok=True)
    df.to_csv(f"{PROJECT_DIR}/data/processed/churn_data.csv", index=False)
    print(f"Data ingested! Shape: {df.shape}")

def run_data_validation():
    import pandas as pd
    df = pd.read_csv(f"{PROJECT_DIR}/data/processed/churn_data.csv")
    assert df.shape[0] > 0
    assert 'Churn' in df.columns
    print(f"Validation passed! Rows: {df.shape[0]}")

def run_feature_engineering():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(f"{PROJECT_DIR}/data/processed/churn_data.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    base = f"{PROJECT_DIR}/data/processed"
    X_train.to_csv(f"{base}/X_train.csv", index=False)
    X_test.to_csv(f"{base}/X_test.csv", index=False)
    y_train.to_csv(f"{base}/y_train.csv", index=False)
    y_test.to_csv(f"{base}/y_test.csv", index=False)
    print("Feature engineering done!")

def run_model_training():
    import subprocess, sys
    result = subprocess.run([sys.executable, f"{PROJECT_DIR}/src/train.py"], cwd=PROJECT_DIR, check=True, capture_output=True, text=True)
    print(result.stdout)

def run_model_evaluation():
    import subprocess, sys
    result = subprocess.run([sys.executable, f"{PROJECT_DIR}/src/evaluate.py"], cwd=PROJECT_DIR, check=True, capture_output=True, text=True)
    print(result.stdout)

def run_model_registration():
    import mlflow
    from mlflow.tracking import MlflowClient
    from dotenv import load_dotenv
    load_dotenv(f"{PROJECT_DIR}/.env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()
    experiment = client.get_experiment_by_name("churn-prediction")
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.roc_auc DESC"],
            max_results=1
        )
        if runs:
            best_run = runs[0]
            model_uri = f"runs:/{best_run.info.run_id}/RandomForest"
            mlflow.register_model(model_uri, "ChurnPredictionModel")
            print(f"Best ROC AUC: {best_run.data.metrics.get('roc_auc')}")
            print("Model registered to MLflow Model Registry successfully!")

task1 = PythonOperator(task_id='data_ingestion', python_callable=run_data_ingestion, dag=dag)
task2 = PythonOperator(task_id='data_validation', python_callable=run_data_validation, dag=dag)
task3 = PythonOperator(task_id='feature_engineering', python_callable=run_feature_engineering, dag=dag)
task4 = PythonOperator(task_id='model_training', python_callable=run_model_training, dag=dag)
task5 = PythonOperator(task_id='model_evaluation', python_callable=run_model_evaluation, dag=dag)
task6 = PythonOperator(task_id='model_registration', python_callable=run_model_registration, dag=dag)

task1 >> task2 >> task3 >> task4 >> task5 >> task6
