# 🧩 MLOps Microservice Project

Ce projet est un **pipeline ML complet** intégré dans un microservice FastAPI.  
Il a été développé dans le cadre du cours **MLOps - Master 2**.

---

## 📂 Structure du projet

project/
│── configs/ # Fichiers de configuration
│── data/ # Données (raw.csv généré automatiquement)
│── ml_microservice/ # Microservice FastAPI (API)
│ ├── app.py # Application FastAPI
│ └── init.py
│── models/ # Modèles sauvegardés
│── src/ # Scripts ML
│ ├── train.py # Entraînement du modèle
│ └── evaluate.py # Évaluation + MLflow
│── tests/ # Tests unitaires & API
│ └── test_app.py
│── Makefile # Automatisation des commandes
│── requirements.txt # Dépendances Python

yaml
Copier le code

---

## ⚙️ Installation

### 1️⃣ Créer l'environnement virtuel
```bash
make init
2️⃣ Activer l'environnement
bash
Copier le code
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
3️⃣ Installer les dépendances (si besoin)
bash
Copier le code
pip install -r requirements.txt
🚀 Utilisation
🏋️‍♂️ Entraîner le modèle
bash
Copier le code
make train
📊 Évaluer le modèle
bash
Copier le code
make evaluate
Les résultats (métriques + courbes) sont loggés dans MLflow.
Lancer l’UI :

bash
Copier le code
mlflow ui
puis aller sur http://127.0.0.1:5000.

🌐 Lancer le microservice API
bash
Copier le code
make run
L’API est dispo sur http://127.0.0.1:8000.

Endpoints disponibles :

GET / → Vérification de l’API

POST /predict → Prédiction à partir d’un JSON de features
Exemple :

json
Copier le code
{
  "features": [14.2, 20.3, 93.5, 600.5, 0.1, ...]
}
🧪 Tests
bash
Copier le code
make test
🐳 Docker (optionnel)
Construire l’image Docker :

bash
Copier le code
make build
