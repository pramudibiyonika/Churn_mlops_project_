from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {type(model).__name__}")
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
    model_used: str

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        df = pd.DataFrame([data.dict()])
        if scaler is not None:
            numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        probability = float(model.predict_proba(df)[0][1])
        prediction = "Yes" if probability >= 0.5 else "No"
        return PredictionResponse(
            churn_probability=round(probability, 4),
            prediction=prediction,
            model_used=type(model).__name__
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
