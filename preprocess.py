import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_monk_dataset(file_path):
    """
    Carica e preprocessa un dataset Monk's.

    Args:
        file_path (str): Percorso del file Monk's.

    Returns:
        np.ndarray: Attributi preprocessati (X).
        np.ndarray: Target (y).
    """
    # Definizione delle colonne del dataset Monk's
    columns = ["class", "a1", "a2", "a3", "a4", "a5", "a6"]

    # Carica il file Monk's
    data = pd.read_csv(file_path, sep='\s+', header=None, names=columns)

    # Separazione degli attributi (X) e del target (y)
    X = data.iloc[:, 1:].values  # Esclude la colonna 'class'
    y = data["class"].values     # Colonna 'class' è il target

    # Preprocessing: One-Hot Encoding per gli attributi discreti
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)

    return X_encoded, y

def load_all_monks(base_path):
    """
    Carica e preprocessa tutti i dataset Monk's (1, 2, 3).

    Args:
        base_path (str): Directory base contenente i file Monk's.

    Returns:
        dict: Dizionario con i dataset preprocessati per Monk's 1, 2 e 3.
    """
    datasets = {}
    for i in range(1, 4):  # Per i tre dataset Monk's
        train_file = f"{base_path}/monks-{i}.train"
        test_file = f"{base_path}/monks-{i}.test"

        X_train, y_train = load_monk_dataset(train_file)
        X_test, y_test = load_monk_dataset(test_file)

        datasets[f"monks-{i}"] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }

    return datasets
