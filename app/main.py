"""
=========================================================
⚡ FastAPI - Heart Disease Prediction API
---------------------------------------------------------
Model: Logistic Regression Pipeline (imputer -> scaler -> log_reg)
Trained on heart.csv, saved as heart_disease_model.joblib
Pydantic models defined in app/schemas.py
---------------------------------------------------------
Endpoints:
GET  /health   -> simple liveness check
GET  /info     -> model type + feature list
POST /predict  -> returns heart_disease true/false + probability
=========================================================
"""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartInput, PredictionOutput, ModelInfo

app = FastAPI(title="Heart Disease Prediction API")

# -----------------------------------------------------------
# Load the trained pipeline once at startup
# -----------------------------------------------------------
MODEL_PATH = Path("model/heart_disease_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Train and save the model first (see training script)."
    )

model = joblib.load(MODEL_PATH)

# Feature order must exactly match what the model was trained on.
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


# ----------------------------------------------------------------
# GET /health -> Health check for load balancers / uptime monitors
# ----------------------------------------------------------------
@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    Load balancers (Nginx, AWS ELB, Kubernetes) call this to verify
    the instance is alive and ready to serve requests.
    """
    return {"status": "healthy"}


# ----------------------------------------------------------------
# GET /info -> Basic model metadata
# ----------------------------------------------------------------
@app.get("/info", response_model=ModelInfo)
def model_info():
    """Returns the model type and the list of features it expects."""
    return {
        "model_type": type(model.named_steps["log_reg"]).__name__,
        "features": FEATURE_NAMES
    }


# ----------------------------------------------------------------
# POST /predict -> Run inference on a single patient's data
# ----------------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict_heart_disease(data: HeartInput):
    """
    Accepts patient feature data, runs it through the saved pipeline
    (imputer -> scaler -> logistic regression), and returns whether
    the model predicts heart disease, along with the probability.
    """
    try:
        # Build a single-row DataFrame with columns in the exact
        # order the model expects (pipeline was fit on a DataFrame,
        # so passing a DataFrame here keeps column names aligned).
        input_df = pd.DataFrame([[
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]], columns=FEATURE_NAMES)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # probability of class 1 (disease)

        return {
            "heart_disease": bool(prediction),
            "probability": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")