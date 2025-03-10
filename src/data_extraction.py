import pandas as pd

def load_data(filepath):
    """Charge les données CSV et retourne un DataFrame."""
    try:
        df = pd.read_csv(filepath)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Le fichier CSV doit contenir les colonnes 'text' et 'label'.")
        return df
    except FileNotFoundError:
        print(f"Erreur : Fichier {filepath} introuvable.")
        return None

if __name__ == "__main__":
    dataset_path = "/Users/utilisateur/Desktop/Projet_remis/dataset_fixed.csv"
    data = load_data(dataset_path)
    if data is not None:
        print(data.head())  # Afficher les premières lignes du dataset
 

import pandas as pd

def load_data(filepath):
    """Charge les données CSV et retourne un DataFrame."""
    try:
        df = pd.read_csv(filepath)
        if "text" not in df.columns:
            raise ValueError("La colonne 'text' est manquante.")
        return df
    except FileNotFoundError:
        print(f"Erreur : Fichier {filepath} introuvable.")
        return None