from fastapi.testclient import TestClient
from app.fastapi_app import app

client = TestClient(app)


# ✅ 1. Health endpoint
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ✅ 2. Valid prediction
def test_valid_prediction():
    payload = {
        "age": 45,
        "gender": "MALE",
        "symptoms": "chest pain, dizziness",
        "symptom_count": 2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()


# ✅ 3. Invalid gender category
def test_invalid_gender_category():
    payload = {
        "age": 30,
        "gender": "unknown",
        "symptoms": "headache",
        "symptom_count": 1,
    }

    response = client.post("/predict", json=payload)

    # Your API currently allows unknown gender → returns prediction
    assert response.status_code == 200
    assert "prediction" in response.json()


# ✅ 4. Missing field
def test_missing_gender():
    payload = {
        "age": 30,
        "symptoms": "headache",
        "symptom_count": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
