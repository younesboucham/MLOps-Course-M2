from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os

app = FastAPI(title="ML Prediction API", version="1.0")

# === Charger le modèle ===
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model_path = os.path.abspath(model_path)

if not os.path.exists(model_path):
    raise RuntimeError(f"❌ Modèle introuvable à {model_path}, lance d'abord `make train`")

model = joblib.load(model_path)


@app.get("/")
def root():
    """Endpoint racine pour vérifier que l’API est en ligne"""
    return {"message": "API de prédiction prête 🚀"}


@app.post("/predict")
def predict(payload: dict):
    """Endpoint de prédiction"""
    features = payload.get("features")
    if features is None:
        raise HTTPException(status_code=400, detail="Champ 'features' manquant")

    try:
        X = np.array(features).reshape(1, -1)
        y_pred = model.predict(X)[0]
        return {"prediction": int(y_pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {str(e)}")
