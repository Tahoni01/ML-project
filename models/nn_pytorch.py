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
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import pandas as pd

results_tr = []
results_ts = []

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_fold(model, x_train, y_train, x_val, y_val, epochs, lr, device, weight_decay):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9 ,weight_decay=weight_decay,nesterov=True)
    model = model.to(device)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())
        train_preds = (outputs.sigmoid() > 0.5).cpu().numpy()
        train_acc_history.append(accuracy_score(y_train, train_preds))

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            val_loss_history.append(val_loss.item())
            val_preds = (val_outputs.sigmoid() > 0.5).cpu().numpy()
            val_acc_history.append(accuracy_score(y_val, val_preds))

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_accuracy": train_acc_history,
        "val_accuracy": val_acc_history,
    }

def k_fold_cross_validation(x, y, input_size, output_size, hidden_size = None, epochs = 100, lr = None, k = 5, device = None, weight_decay = None, model = None):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    val_accuracies = []
    all_metrics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for train_idx, val_idx in skf.split(x,y):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model is None:
            model = Net(input_size, hidden_size, output_size)

        metrics = train_fold(
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

def optimize_hyperparameters(x, y, input_size, output_size, param_bounds, epochs, k, device):
    def objective(hidden_size, lr, weight_decay):
        val_accuracy_mean, _ = k_fold_cross_validation(x, y, input_size, output_size, int(hidden_size), 100, lr, k, device, weight_decay)
        return val_accuracy_mean

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=param_bounds,
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=10)

    best_params = optimizer.max["params"]
    best_params["hidden_size"] = int(best_params["hidden_size"])

    return best_params

def train_model(x, y, input_size, output_size, best_params, epochs, database, device):
    model = Net(input_size, best_params["hidden_size"], output_size)
    _, avg = k_fold_cross_validation(
        x, y, input_size, output_size,
        hidden_size=best_params["hidden_size"],
        epochs=epochs,
        lr=best_params["lr"],
        k=5,
        device=device,
        weight_decay=best_params["weight_decay"],
        model=model
    )

    plot_learning_curve(avg,epochs,database)

    model.eval()
    with torch.no_grad():
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        accuracy = accuracy_score(y.cpu(), preds.cpu())
        mse = mean_squared_error(y.cpu(), preds.cpu())

    results = {
        "Dataset": database,
        "MSE": mse,
        "Accuracy": accuracy
    }
    return model, results

def evaluation(model, x_ts, y_ts, database, device):
    x_ts = torch.tensor(x_ts, dtype=torch.float32).to(device)
    y_ts = torch.tensor(y_ts, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x_ts)
        _, preds = torch.max(outputs, 1)
        accuracy = accuracy_score(y_ts.cpu(), preds.cpu())
        mse = mean_squared_error(y_ts.cpu(), preds.cpu())

    results = {
        "Dataset": database,
        "MSE": mse,
        "Accuracy": accuracy
    }
    return results

def plot_learning_curve(metrics, epochs, database):
    # Crea una sequenza per le epoche
    epoch_range = list(range(1, epochs + 1))  # [1, 2, ..., epochs]

    # Plot delle loss
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, metrics["train_loss"], label="Train Loss")
    plt.plot(epoch_range, metrics["val_loss"], label="Validation Loss", linestyle="--")
    plt.title(f"Average Loss Curves {database}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    file_name = f"results/loss_curve_{database}.png"
    plt.savefig(file_name)
    plt.close()

    # Plot delle accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, metrics["train_accuracy"], label="Train Accuracy")
    plt.plot(epoch_range, metrics["val_accuracy"], label="Validation Accuracy", linestyle="--")
    plt.title(f"Average Accuracy Curves {database}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    file_name = f"results/accuracy_curve_{database}.png"
    plt.savefig(file_name)
    plt.close()

def plot_results_nn():
    merged_df = merge_results(results_tr,results_ts)
    fig, ax = plt.subplots(figsize=(10, len(merged_df) * 0.6))  # Altezza dinamica in base al numero di righe
    ax.axis('tight')
    ax.axis('off')

    # Crea la tabella
    table = ax.table(cellText=merged_df.values, colLabels=merged_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(merged_df.columns))))

    # Salva il grafico come immagine
    plt.title("Risultati per Dataset")
    file_name = f"results/results_nn.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Tabella salvata")

def merge_results(results_train, results_test):
    # Converti le liste di dizionari in DataFrame
    df_train = pd.DataFrame(results_train)
    df_test = pd.DataFrame(results_test)

    # Rinomina le colonne di train e test per differenziarle
    df_train.rename(columns={"MSE": "MSE (TR)", "Accuracy": "Accuracy (TR)"}, inplace=True)
    df_test.rename(columns={"MSE": "MSE (TS)", "Accuracy": "Accuracy (TS)"}, inplace=True)

    # Unisci i due DataFrame sulla colonna "Dataset"
    merged_df = pd.merge(df_train, df_test, on="Dataset", how="outer")

    # Ordina i risultati per Dataset
    merged_df.sort_values(by="Dataset", inplace=True)

    # Approssima i valori a 10^-4
    merged_df = merged_df.round(4)

    # Riordina le colonne nel formato richiesto
    column_order = ["Dataset", "MSE (TR)", "MSE (TS)", "Accuracy (TR)", "Accuracy (TS)"]
    merged_df = merged_df[column_order]

    return merged_df

def nn_execution(x_tr, y_tr, x_ts, y_ts, input_size, output_size, database):
    param_bounds = {
        "hidden_size": (16, 100),
        "lr": (0.005, 0.5),
        "weight_decay": (0.0001, 0.01),
    }
    epochs = 500
    k = 5
    device = "cpu"

    best_params = optimize_hyperparameters(x_tr, y_tr, input_size,output_size ,param_bounds, epochs, k, device)

    model, train_results = train_model(x_tr, y_tr, input_size, output_size ,best_params, epochs, database, device)
    test_results = evaluation(model, x_ts, y_ts, database, device)

    results_tr.append(train_results)
    results_ts.append(test_results)
