import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from api import app 

VALID_CLIENT = {
    "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 1000, "BILL_AMT2": 1000, "BILL_AMT3": 1000, "BILL_AMT4": 1000, "BILL_AMT5": 1000, "BILL_AMT6": 1000,
    "PAY_AMT1": 500, "PAY_AMT2": 500, "PAY_AMT3": 500, "PAY_AMT4": 500, "PAY_AMT5": 500, "PAY_AMT6": 500,
}

@pytest.fixture
def client():
    """Тестовый клиент Flask"""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
        
def test_health(client):
    """GET /health, возвращает 200 и статус healthy"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    
def test_predict_valid(client):
    """POST /predict с корректными данными, возвращает прогноз и вероятность"""
    resp = client.post("/predict", json=VALID_CLIENT)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    
def test_predict_explicit_version(client):
    """POST /predict с явной версией"""
    payload = dict(VALID_CLIENT, model_version="v2")
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["model_version"] == "v2"
    assert data["routing"] == "explicit"
    
def test_predict_ab_routing(client):
    """POST /predict без версии использует A/B-роутинг"""
    resp = client.post("/predict", json=VALID_CLIENT)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["routing"] == "ab_random"
    assert data["model_version"] in ("v1", "v2")
    
def test_predict_missing_features(client):
    """POST /predict с нехваткой признаков, возвращает 400"""
    resp = client.post("/predict", json={"LIMIT_BAL": 50000})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
    
def test_predict_empty_body(client):
    """POST /predict с пустым телом, возвращает ошибку"""
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
