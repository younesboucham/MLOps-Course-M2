import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

# Charger un dataset jouet
data = load_breast_cancer(as_frame=True)
df = data.frame  # contient déjà la colonne "target"

# Créer le dossier data/ si besoin
os.makedirs("../data", exist_ok=True)

# Sauvegarder en CSV (dans project/data/raw.csv)
csv_path = os.path.join("..", "data", "raw.csv")
df.to_csv(csv_path, index=False)

# Vérification
print(f"✅ Dataset généré dans {csv_path} avec {df.shape[0]} lignes et {df.shape[1]} colonnes")
