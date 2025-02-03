import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_monk_data(file_path, encoder=None, scaler=None):
    #Load and preprocess a MONK dataset from a file.

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File {file_path} does not exist.")

    # Read the dataset
    data = pd.read_csv(file_path, sep='\s+', header=None)

    # Extract input features (X) and target labels (Y)
    X = data.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    Y = data.iloc[:, 0].to_numpy(dtype=np.float32).ravel()

    # Apply one-hot encoding if no encoder is provided
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
        X = encoder.fit_transform(X)
    else:
        X = encoder.transform(X)

    # Apply standard scaling if no scaler is provided
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # Convert data to PyTorch tensors
    X_torch = torch.from_numpy(X)
    Y_torch = torch.from_numpy(Y).unsqueeze(1)

    return {"numpy": (X, Y), "torch": (X_torch, Y_torch)}, encoder, scaler

def load_all_monks(base_path):
    #Load all MONK datasets (monks-1, monks-2, monks-3) and preprocess them.

    datasets = {}
    encoders = {}
    scalers = {}

    for monk_name in ["monks-1", "monks-2", "monks-3"]:
        train_path = os.path.join(base_path, f"{monk_name}.train")
        test_path = os.path.join(base_path, f"{monk_name}.test")

        train_data, encoder, scaler = load_monk_data(train_path)

        test_data, _, _ = load_monk_data(test_path, encoder=encoder, scaler=scaler)

        datasets[monk_name] = {"train": train_data, "test": test_data}
        encoders[monk_name] = encoder
        scalers[monk_name] = scaler

    return datasets, encoders, scalers

def load_cup_data(train_path, test_path):
    #Load and preprocess the ML-CUP dataset (train and test).

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("❌ Train or test file is missing!")

    # Read training and test data, skipping the first 7 rows (headers)
    df_train = pd.read_csv(train_path, skiprows=7, header=None)
    df_test = pd.read_csv(test_path, skiprows=7, header=None)

    # Extract features and target values
    X_train = df_train.iloc[:, 1:-3].to_numpy(dtype=np.float32)  # Features
    Y_train = df_train.iloc[:, -3:].to_numpy(dtype=np.float32)  # Targets
    X_test = df_test.iloc[:, 1:].to_numpy(dtype=np.float32)  # Test features

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_torch = torch.from_numpy(X_train)
    Y_train_torch = torch.from_numpy(Y_train)
    X_test_torch = torch.from_numpy(X_test)

    return {
        "train": {"numpy": (X_train, Y_train), "torch": (X_train_torch, Y_train_torch)},
        "test": {"numpy": X_test, "torch": X_test_torch},
    }
