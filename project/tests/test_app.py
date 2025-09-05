from fastapi.testclient import TestClient
from ml_microservice.app import app

client = TestClient(app)


def test_root():
    """Test que l’API répond sur /"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_valid():
    """Test une requête valide sur /predict"""
    # Exemple avec 30 features du dataset Breast Cancer
    features = [14.2, 20.3, 93.5, 600.5, 0.1, 0.2, 0.3, 0.1, 0.2, 0.06,
                0.5, 1.2, 3.2, 40.5, 0.006, 0.02, 0.03, 0.01, 0.01, 0.002,
                16.0, 25.3, 105.5, 800.0, 0.12, 0.25, 0.35, 0.15, 0.20, 0.07]

    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]


def test_predict_invalid():
    """Test une requête invalide (pas assez de features)"""
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code in (400, 422)
