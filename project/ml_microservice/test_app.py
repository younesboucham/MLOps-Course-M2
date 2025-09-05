from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Charger le modèle
model = joblib.load("../models/model.pkl")

@app.post("/predict")
def predict(features: list[float]):
    X = np.array(features).reshape(1, -1)
    y_pred = model.predict(X)[0]
    return {"prediction": int(y_pred)}
