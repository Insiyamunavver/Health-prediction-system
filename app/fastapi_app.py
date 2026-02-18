from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# 🚀 Create FastAPI app
app = FastAPI(title="Healthcare Disease Prediction API")

# 📦 Load trained pipeline

print("Loading model from:", os.path.abspath("model.joblib"))

model = joblib.load("model.joblib")
print(model)


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
