"""
import pandas as pd
import numpy as np
import torch

def load_monk_data(file_path):
    # Legge i dati separati da spazi
    data = pd.read_csv(file_path, sep='\s+', header=None)

    # Separa le feature (prime 6 colonne) e il target (ultima colonna)
    X = data.iloc[:, 1:-1].values.astype(np.float32)  # Salta 'Id' e 'class'
    Y = data.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)  # 'class' come 2D

    # Converti in Tensors per PyTorch
    X_torch = torch.tensor(X)
    Y_torch = torch.tensor(Y)

    return {"numpy": (X, Y), "torch": (X_torch, Y_torch)}


def load_all_monks(base_path):
    datasets = {}
    for monk_name in ["monks-1", "monks-2", "monks-3"]:
        # Percorsi per i file train e test
        train_path = f"{base_path}/{monk_name}.train"
        test_path = f"{base_path}/{monk_name}.test"

        # Caricamento dei dati di training e test
        train_data = load_monk_data(train_path)
        test_data = load_monk_data(test_path)

        # Organizzazione dei dati nel dizionario
        datasets[monk_name] = {
            "train": train_data,
            "test": test_data,
        }
    return datasets


def load_cup_data(train_path, test_path):
    # Caricamento del file di training
    df_train = pd.read_csv(train_path, skiprows=7, header=None)
    df_train_filtered = df_train.iloc[:, 1:16]  # Colonne da 'INPUTS' fino a 'P'

    # Dividi X (input) e Y (target) per il training
    X_train = df_train_filtered.iloc[:, :-3].values.astype(np.float32)  # Tutte le colonne tranne le ultime 3
    Y_train = df_train_filtered.iloc[:, -3:].values.astype(np.float32)  # Le ultime 3 colonne

    # Caricamento del file di test
    df_test = pd.read_csv(test_path, skiprows=7, header=None)
    df_test_filtered = df_test.iloc[:, 1:16]  # Colonne da 'INPUTS' fino a 'P'

    # Dividi X (input) e Y (target) per il test
    X_test = df_test_filtered.iloc[:, :-3].values.astype(np.float32)
    Y_test = df_test_filtered.iloc[:, -3:].values.astype(np.float32)

    # Converti in Tensors per PyTorch
    X_train_torch = torch.tensor(X_train)
    Y_train_torch = torch.tensor(Y_train)
    X_test_torch = torch.tensor(X_test)
    Y_test_torch = torch.tensor(Y_test)

    return {
        "train": {"numpy": (X_train, Y_train), "torch": (X_train_torch, Y_train_torch)},
        "test": {"numpy": (X_test, Y_test), "torch": (X_test_torch, Y_test_torch)},
    }
"""



import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

"""
def load_monk_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Il file {file_path} non esiste.")

    data = pd.read_csv(file_path, sep='\s+', header=None)

    # Separa features e target
    X = data.iloc[:, 1:-1].to_numpy(dtype=np.float32)  # Esclude 'Id' e 'class'
    Y = data.iloc[:, 0].to_numpy(dtype=np.float32).ravel()  # 🔹 Converte in 1D 

    # Converti in Tensors
    X_torch = torch.from_numpy(X)
    Y_torch = torch.from_numpy(Y).unsqueeze(1)

    return {"numpy": (X, Y), "torch": (X_torch, Y_torch)}

def load_all_monks(base_path):
    datasets = {}
    for monk_name in ["monks-1", "monks-2", "monks-3"]:
        train_path = os.path.join(base_path, f"{monk_name}.train")
        test_path = os.path.join(base_path, f"{monk_name}.test")

        train_data = load_monk_data(train_path)
        test_data = load_monk_data(test_path)

        datasets[monk_name] = {"train": train_data, "test": test_data}
    return datasets
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_monk_data(file_path, encoder=None, scaler=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"\u274c Il file {file_path} non esiste.")

    data = pd.read_csv(file_path, sep='\s+', header=None)

    # Separa features e target
    X = data.iloc[:, 1:-1].to_numpy(dtype=np.float32)  # Esclude 'Id' e 'class'
    Y = data.iloc[:, 0].to_numpy(dtype=np.float32).ravel()  # 🔹 Converte in 1D

    # Preprocessing con One-Hot Encoding e StandardScaler
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
        X = encoder.fit_transform(X)
    else:
        X = encoder.transform(X)

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # Converti in Tensors
    X_torch = torch.from_numpy(X)
    Y_torch = torch.from_numpy(Y).unsqueeze(1)

    return {"numpy": (X, Y), "torch": (X_torch, Y_torch)}, encoder, scaler

def load_all_monks(base_path):
    datasets = {}
    encoders = {}
    scalers = {}

    for monk_name in ["monks-1", "monks-2", "monks-3"]:
        train_path = os.path.join(base_path, f"{monk_name}.train")
        test_path = os.path.join(base_path, f"{monk_name}.test")

        # Carica e preprocessa i dati di train
        train_data, encoder, scaler = load_monk_data(train_path)

        # Usa lo stesso encoder e scaler per il test
        test_data, _, _ = load_monk_data(test_path, encoder=encoder, scaler=scaler)

        datasets[monk_name] = {"train": train_data, "test": test_data}
        encoders[monk_name] = encoder
        scalers[monk_name] = scaler

    return datasets, encoders, scalers

"""
def load_cup_data(train_path, test_path):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("File train o test mancanti!")

    df_train = pd.read_csv(train_path, skiprows=7, header=None)
    df_test = pd.read_csv(test_path, skiprows=7, header=None)

    # Seleziona le colonne corrette
    X_train = df_train.iloc[:, 1:-3].to_numpy(dtype=np.float32)  # Feature
    Y_train = df_train.iloc[:, -3:].to_numpy(dtype=np.float32)
    X_test = df_test.iloc[:, 1:].to_numpy(dtype=np.float32)

    # Converti in Tensors
    X_train_torch = torch.from_numpy(X_train)
    Y_train_torch = torch.from_numpy(Y_train)
    X_test_torch = torch.from_numpy(X_test)

    return {
        "train": {"numpy": (X_train, Y_train), "torch": (X_train_torch, Y_train_torch)},
        "test": {"numpy": X_test, "torch": X_test_torch},
    }
"""
def load_cup_data(train_path, test_path):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("File train o test mancanti!")

    df_train = pd.read_csv(train_path, skiprows=7, header=None)
    df_test = pd.read_csv(test_path, skiprows=7, header=None)

    # Seleziona le colonne corrette
    X_train = df_train.iloc[:, 1:-3].to_numpy(dtype=np.float32)  # Feature
    Y_train = df_train.iloc[:, -3:].to_numpy(dtype=np.float32)  # Target
    X_test = df_test.iloc[:, 1:].to_numpy(dtype=np.float32)  # Feature test

    # 🔹 Normalizzazione delle feature con StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Adatta sui dati di train e trasforma
    X_test = scaler.transform(X_test)  # Trasforma solo, senza adattare

    # Converti in Tensors
    X_train_torch = torch.from_numpy(X_train)
    Y_train_torch = torch.from_numpy(Y_train)
    X_test_torch = torch.from_numpy(X_test)

    return {
        "train": {"numpy": (X_train, Y_train), "torch": (X_train_torch, Y_train_torch)},
        "test": {"numpy": X_test, "torch": X_test_torch},
    }
