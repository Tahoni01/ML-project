"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from bayes_opt import BayesianOptimization
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd

# PyTorch Model for Classification
class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

def train_model(model, x_train, y_train, x_val, y_val, epochs, lr, device, patience, weight_decay):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []


    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())
        train_preds = (outputs.sigmoid() > 0.5).cpu().numpy()
        train_acc_history.append(accuracy_score(y_train, train_preds))

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_loss_history.append(val_loss.item())
            val_preds = (val_outputs.sigmoid() > 0.5).cpu().numpy()
            val_acc_history.append(accuracy_score(y_val, val_preds))

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_accuracy": train_acc_history,
        "val_accuracy": val_acc_history,
    }

def k_fold_cross_validation(x, y, input_size, hidden_size, epochs, lr, k, device, weight_decay, patience=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_accuracies = []
    all_metrics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for train_idx, val_idx in kf.split(x):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = PyTorchNN(input_size, hidden_size)
        metrics = train_model(
            model, x_train, y_train, x_val, y_val, epochs, lr, device, patience, weight_decay
        )

        val_accuracies.append(np.mean(metrics["val_accuracy"]))
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    # Compute averages
    avg_metrics = {
        key: np.nanmean([np.pad(m, (0, max(map(len, all_metrics[key])) - len(m)), constant_values=np.nan) for m in all_metrics[key]], axis=0).tolist()
        for key in all_metrics
    }
    return np.mean(val_accuracies), avg_metrics

def optimize_hyperparameters(x, y, input_size, param_bounds, epochs, k, device):
    def objective(hidden_size, lr, weight_decay):
        return k_fold_cross_validation(
            x, y, input_size, int(hidden_size), epochs, lr, k, device, weight_decay, patience=10
        )[0]

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=param_bounds,
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=20)

    best_params = optimizer.max["params"]
    best_params["hidden_size"] = int(best_params["hidden_size"])
    best_val_accuracy = optimizer.max["target"]  # Ottieni il valore migliore

    final_model = PyTorchNN(input_size, best_params["hidden_size"])
    _, avg_metrics = k_fold_cross_validation(
        x, y, input_size, best_params["hidden_size"], epochs, best_params["lr"], k, device, best_params["weight_decay"], patience=10
    )

    plot_combined_metrics(avg_metrics)
    return final_model, best_params, best_val_accuracy

def plot_combined_metrics(metrics, result_dir="results"):
    os.makedirs(result_dir, exist_ok=True)
    epochs = range(1, len(metrics["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss", linestyle="--")
    plt.title("Average Loss Curves Across Folds")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{result_dir}/average_loss_curves.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, metrics["val_accuracy"], label="Validation Accuracy", linestyle="--")
    plt.title("Average Accuracy Curves Across Folds")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{result_dir}/average_accuracy_curves.png")
    plt.close()

def mean_euclidean_error(predictions, targets):
    predictions = np.array(predictions).reshape(-1, 1)
    targets = np.array(targets).reshape(-1, 1)
    return np.mean(np.sqrt((predictions - targets) ** 2))


def final_evaluation(model, X_train, y_train, X_test, y_test, device="cpu"):
    model = model.to(device)

    # Converte i dati in tensori PyTorch
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        # Predizioni continue per il training set
        train_outputs = model(x_train_tensor).cpu().numpy().squeeze()  # Ottieni valori continui
        train_preds = (train_outputs > 0).astype(int)  # Converti in 0/1
        train_accuracy = accuracy_score(y_train, train_preds)
        train_loss = mean_euclidean_error(train_outputs, y_train)  # Usa valori continui per l'MEE

        # Predizioni continue per il test set
        test_outputs = model(x_test_tensor).cpu().numpy().squeeze()  # Ottieni valori continui
        test_preds = (test_outputs > 0).astype(int)  # Converti in 0/1
        test_accuracy = accuracy_score(y_test, test_preds)
        test_loss = mean_euclidean_error(test_outputs, y_test)  # Usa valori continui per l'MEE

    # Tabella dei risultati
    result_data = {
        "Task": ["Monk"],
        "MEE (TR)": [train_loss],
        "MEE (TS)": [test_loss],
        "Accuracy (TR)": [train_accuracy],
        "Accuracy (TS)": [test_accuracy],
    }
    results_df = pd.DataFrame(result_data)
    results_df.to_csv("results/final_results.csv", index=False)

    print("Results saved in the 'results' directory.")
    return test_accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from bayes_opt import BayesianOptimization
import os
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd

# PyTorch Model for Classification
class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size,1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_model(model, X_train, y_train, X_val, y_val, epochs, lr, device, weight_decay):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)  # Fit e trasformazione sui dati di training
    x_val = scaler.transform(X_val)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight_tensor[1])  # Solo per la classe positiva

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9 ,weight_decay=weight_decay,nesterov=True)
    model = model.to(device)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())
        train_preds = (outputs.sigmoid() > 0.5).cpu().numpy()
        train_acc_history.append(accuracy_score(y_train, train_preds))

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_loss_history.append(val_loss.item())
            val_preds = (val_outputs.sigmoid() > 0.5).cpu().numpy()
            val_acc_history.append(accuracy_score(y_val, val_preds))

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_accuracy": train_acc_history,
        "val_accuracy": val_acc_history,
    }

def k_fold_cross_validation(x, y, input_size, hidden_size, epochs, lr, k, device, weight_decay):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_accuracies = []
    all_metrics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for train_idx, val_idx in kf.split(x):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = PyTorchNN(input_size, hidden_size)
        metrics = train_model(
            model, x_train, y_train, x_val, y_val, epochs, lr, device, weight_decay
        )

        val_accuracies.append(np.mean(metrics["val_accuracy"]))
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    # Compute averages
    avg_metrics = {
        key: np.nanmean([np.pad(m, (0, max(map(len, all_metrics[key])) - len(m)), constant_values=np.nan) for m in all_metrics[key]], axis=0).tolist()
        for key in all_metrics
    }
    return np.mean(val_accuracies), avg_metrics

def optimize_hyperparameters(x, y, input_size, param_bounds, epochs, k, device, dataset_name):
    def objective(hidden_size, lr, weight_decay):
        return k_fold_cross_validation(
            x, y, input_size, int(hidden_size), epochs, lr, k, device, weight_decay
        )[0]

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=param_bounds,
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=3, n_iter=10)

    best_params = optimizer.max["params"]
    best_params["hidden_size"] = int(best_params["hidden_size"])
    best_val_accuracy = optimizer.max["target"]  # Ottieni il valore migliore

    final_model = PyTorchNN(input_size, best_params["hidden_size"])
    _, avg_metrics = k_fold_cross_validation(
        x, y, input_size, best_params["hidden_size"], epochs, best_params["lr"], k, device, best_params["weight_decay"]
    )

    # Usa il nome del dataset come prefisso per i file
    plot_combined_metrics(avg_metrics, result_dir="results", prefix=f"{dataset_name}_train_validation")
    return final_model, best_params, best_val_accuracy

def plot_combined_metrics(metrics, result_dir="results", prefix="default"):
    os.makedirs(result_dir, exist_ok=True)
    epochs = range(1, len(metrics["train_loss"]) + 1)

    # Plot delle loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss", linestyle="--")
    plt.title(f"Average Loss Curves ({prefix})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{result_dir}/{prefix}_loss_curves.png")
    plt.close()

    # Plot delle accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, metrics["val_accuracy"], label="Validation Accuracy", linestyle="--")
    plt.title(f"Average Accuracy Curves ({prefix})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{result_dir}/{prefix}_accuracy_curves.png")
    plt.close()

def mean_squared_error(y_pred, y_true):
    mse_loss_fn = torch.nn.MSELoss()
    return mse_loss_fn(y_pred, y_true)

def final_evaluation(model, X_train, y_train, X_test, y_test, device="cpu", dataset_name="default"):
    model = model.to(device)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converte i dati in tensori PyTorch
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    y_train_tensor = (y_train_tensor > 0).int()
    y_test_tensor = (y_test_tensor > 0).int()

    model.eval()
    with torch.no_grad():
        # Predizioni continue per il training set
        train_outputs = model(x_train_tensor).sigmoid()
        # Predizioni con la soglia ottimale
        train_preds = (train_outputs > 0.5).float()
        # Calcolo del train_loss
        train_loss = mean_squared_error(train_outputs, y_train_tensor)


        # Predizioni continue per il test set
        test_outputs = model(x_test_tensor).sigmoid()
        test_preds = (test_outputs > 0.5).float()
        # Calcolo del test_loss
        test_loss = mean_squared_error(test_outputs, y_test_tensor)

        # Accuratezza
        train_accuracy = accuracy_score(y_train_tensor.cpu(), train_preds.cpu())
        test_accuracy = accuracy_score(y_test_tensor.cpu(), test_preds.cpu())

    # Tabella dei risultati
    result_data = {
        "Task": [dataset_name],
        "MSE (TR)": [train_loss.item()],
        "MSE (TS)": [test_loss.item()],
        "Accuracy (TR)": [train_accuracy],
        "Accuracy (TS)": [test_accuracy],
    }
    results_df = pd.DataFrame(result_data)
    results_df.to_csv(f"results/{dataset_name}_final_results.csv", index=False)

    print(results_df)