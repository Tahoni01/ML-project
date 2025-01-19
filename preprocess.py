import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_monk_dataset(file_path, encoder=None):
    """
    Carica e preprocessa un singolo dataset MONK.
    Args:
        file_path (str): Percorso al file.
        encoder (OneHotEncoder, opzionale): Encoder da riutilizzare.
    Returns:
        tuple: (X, y, encoder)
    """
    # Colonne del file MONK
    columns = ["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    data = pd.read_csv(file_path, sep='\s+', header=None, names=columns)

    # Separare attributi e target
    X = data.iloc[:, 1:7].values  # Colonne a1, ..., a6
    y = data["class"].values      # Colonna class

    # One-Hot Encoding degli attributi
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X)
    else:
        X_encoded = encoder.transform(X)

    return X_encoded, y, encoder

def load_all_monks(base_path):
    """
    Carica tutti i dataset MONK.
    Args:
        base_path (str): Directory contenente i file MONK.
    Returns:
        dict: Dizionario con i dati di training e utility per ogni MONK.
    """
    datasets = {}
    encoder = None  # Encoder condiviso tra training e utility set
    for i in range(1, 4):  # MONK-1, MONK-2, MONK-3
        train_file = f"{base_path}/monks-{i}.train"
        test_file = f"{base_path}/monks-{i}.test"

        # Preprocessing
        X_train, y_train, encoder = load_monk_dataset(train_file, encoder=encoder)
        X_test, y_test, _ = load_monk_dataset(test_file, encoder=encoder)

        # Memorizza i dati
        datasets[f"monks-{i}"] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }

    return datasets
