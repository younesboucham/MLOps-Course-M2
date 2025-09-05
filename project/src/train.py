import os
import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# === Charger la configuration ===
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
CONFIG_PATH = os.path.abspath(CONFIG_PATH)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# === Construire les chemins ===
project_root = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_root, config["data_path"])
model_dir = os.path.join(project_root, config["model_dir"])
model_path = os.path.join(model_dir, config["model_name"])

# === Charger les données ===
df = pd.read_csv(data_path, header=0)
X = df.drop("target", axis=1)
y = df["target"]

# === Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["train"]["test_size"],
    random_state=config["train"]["random_state"]
)

# === Définir modèle ===
model = LogisticRegression(**config["model"]["params"])

# === GridSearchCV ===
param_grid = {f"{k}": v for k, v in config["gridsearch"]["param_grid"].items()}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=config["gridsearch"]["cv"],
    scoring=config["gridsearch"]["scoring"]
)

# === MLflow setup ===
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

with mlflow.start_run():
    mlflow.sklearn.autolog()  # active autologging pour sklearn

    # Fit
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Logger les meilleurs paramètres manuellement aussi
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # Sauvegarde du modèle final
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, model_path)

    print("✅ Meilleurs paramètres :", grid_search.best_params_)
    print("✅ Score CV :", grid_search.best_score_)
    print(f"✅ Modèle sauvegardé dans {model_path}")
