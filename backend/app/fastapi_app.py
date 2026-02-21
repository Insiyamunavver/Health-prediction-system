from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

import joblib
import pandas as pd
import os

# 🚀 Create FastAPI app
app = FastAPI(title="Healthcare Disease Prediction API")

# 📦 Build correct path to models folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.joblib")

print("Loading model from:", MODEL_PATH)

model = None

# ✅ Safe model loading (CI/CD friendly)
if os.path.exists(MODEL_PATH):
    print(f"✅ Model found. Loading {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
else:
    print("⚠ model.joblib NOT found (running in test/CI mode)")


# 📊 Prometheus Metrics
REQUEST_COUNT = Counter("request_count", "Total prediction requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Prediction latency")


# 🧾 Pydantic Request Model
class PatientData(BaseModel):
    age: int
    gender: str
    symptoms: str
    symptom_count: int


# 🧾 Pydantic Response Model
class PredictionResponse(BaseModel):
    prediction: str


# ❤️ Health Check Endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


# 🔮 Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):

    REQUEST_COUNT.inc()

    # ❗ Protect against missing model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    with REQUEST_LATENCY.time():

        # Convert input → DataFrame
        input_df = pd.DataFrame([{
            "age": data.age,
            "gender": data.gender,
            "symptoms": data.symptoms,
            "symptom_count": data.symptom_count
        }])

        prediction = model.predict(input_df)[0]

    return PredictionResponse(prediction=prediction)


# 📈 Prometheus Metrics Endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")