import pandas as pd
import numpy as np
import torch

def load_monk_data(file_path):
    # Legge i dati separati da spazi
    data = pd.read_csv(file_path, sep='\s+', header=None)

    # Separa le feature (prime 6 colonne) e il target (ultima colonna)
    X = data.iloc[:, 1:-1].values.astype(np.float32)  # Salta 'Id' e 'class'
    Y = data.iloc[:, 0].values.astype(np.float32)  # 'class'

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