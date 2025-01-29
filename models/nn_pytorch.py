import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

results_tr = []
results_ts = []

param_bounds = {
    'hidden_size': (2, 128),
    'lr': (0.009, 0.9),
    'weight_decay': (0.0001, 0.001)
}
epochs = 500
k = 5
device = "cpu"

def mee(y_true, y_pred): return torch.mean(torch.norm(y_pred - y_true, dim=1))

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, task):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(hidden_size, output_size)
        if task == "classification":
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


def train_fold(model, x_train, y_train, x_val, y_val, epochs, lr, device, weight_decay):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
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
        train_preds = (outputs > 0.5).cpu().numpy()
        train_acc_history.append(accuracy_score(y_train, train_preds))

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            val_loss_history.append(val_loss.item())
            val_preds = (val_outputs > 0.5).cpu().numpy()
            val_acc_history.append(accuracy_score(y_val, val_preds))

    return {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_accuracy": train_acc_history,
        "val_accuracy": val_acc_history,
    }


def k_fold_cross_validation(x, y, input_size, output_size, hidden_size, epochs, lr, k, device, weight_decay):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_accuracies = []
    all_metrics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    model_weights = []

    for train_idx, val_idx in kf.split(x, y):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Net(input_size, hidden_size, output_size,"classification").to(device)

        metrics = train_fold(model, x_train, y_train, x_val, y_val, epochs, lr, device, weight_decay)
        model_weights.append(model.state_dict())  # Salva i pesi del modello

        val_accuracies.append(np.mean(metrics["val_accuracy"]))
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    avg_metrics = {
        key: np.nanmean([np.pad(m, (0, max(map(len, all_metrics[key])) - len(m)), constant_values=np.nan) for m in all_metrics[key]], axis=0).tolist()
        for key in all_metrics
    }

    # Media dei pesi dei modelli addestrati
    averaged_weights = {k: sum(w[k] for w in model_weights) / len(model_weights) for k in model_weights[0]}

    return np.mean(val_accuracies), avg_metrics, averaged_weights


def optimize_hyperparameters(x, y, input_size, output_size, param_bounds, epochs, k, device, task ):
    best_score = 0
    best_params = None
    avg_metrics = None
    averaged_weights = None

    for _ in range(5):  # Random Search con 20 tentativi
        hidden_size = random.randint(*param_bounds["hidden_size"])
        lr = random.uniform(*param_bounds["lr"])
        weight_decay = random.uniform(*param_bounds["weight_decay"])

        val_accuracy_mean, metrics, weights = k_fold_cross_validation(
            x, y, input_size, output_size, hidden_size, epochs, lr, k, device, weight_decay
        )

        if val_accuracy_mean > best_score:
            best_score = val_accuracy_mean
            best_params = {"hidden_size": hidden_size, "lr": lr, "weight_decay": weight_decay}
            avg_metrics = metrics
            averaged_weights = weights

    # Crea il modello migliore con i pesi medi
    best_model = Net(input_size, best_params["hidden_size"], output_size,"classification").to(device)
    best_model.load_state_dict(averaged_weights)

    print(f"Migliori parametri trovati: {best_params}")
    return best_model, best_params, avg_metrics


def evaluation_tr(model, x, y, dataset_name, device=None, task=None):
    model.eval()

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        x_tr, x_vl, y_tr, y_vl = train_test_split(x, y, test_size=0.2, random_state=42)

        y_pred_tr = model(x_tr)
        y_pred_vl = model(x_vl)

        if task == "classification":
            y_pred_tr = (y_pred_tr > 0.5).float()

            accuracy_tr = accuracy_score(y_tr.cpu(), y_pred_tr.cpu())
            mse_tr = mean_squared_error(y_tr.cpu(), y_pred_tr.cpu())

            results = {
                "Dataset": dataset_name,
                "MSE (TR)": mse_tr,
                "Accuracy (TR)": accuracy_tr,
            }

        elif task == "regression":
            mee_tr = mee(y_tr.cpu(), y_pred_tr.cpu())
            mee_vl = mee(y_vl.cpu(), y_pred_vl.cpu())
            results = {
                "Dataset": dataset_name,
                "MEE (TR)": mee_tr,
                "MEE (VL)": mee_vl
            }
    return results

def evaluation_ts(model, x, y, dataset_name, device = None, task = None):
    model.eval()

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        y_pred_ts = model(x)

        if task == "classification":
            y_pred_ts = (y_pred_ts > 0.5).float()

            accuracy_ts = accuracy_score(y.cpu(), y_pred_ts.cpu())
            mse_ts = mean_squared_error(y.cpu(), y_pred_ts.cpu())
            results = {
                "Dataset": dataset_name,
                "MSE (TS)": mse_ts,
                "Accuracy (TS)": accuracy_ts
            }

        elif task == "regression":
            mee_ts = mee(y.cpu(), y_pred_ts.cpu())
            results = {
                "Dataset": dataset_name,
                "MEE (TS)": mee_ts,
            }
    return results

def plot_classification_metrics(metrics, epochs, dataset_name):
    # Verifica che ci siano valori validi in metrics
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    train_accuracy = metrics["train_accuracy"]
    val_accuracy = metrics["val_accuracy"]

    epoch_range = np.arange(1, len(train_loss) + 1)  # La lunghezza di train_loss indica il numero di epoche
    # Plot della Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_loss, label="Train Loss")
    plt.plot(epoch_range, val_loss, label="Validation Loss", linestyle="--")
    plt.title(f"Loss Curve - {dataset_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/loss_curve_{dataset_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Plot dell'Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_accuracy, label="Train Accuracy")
    plt.plot(epoch_range, val_accuracy, label="Validation Accuracy", linestyle="--")
    plt.title(f"Accuracy Curve - {dataset_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/accuracy_curve_{dataset_name}.png", bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Learning curves salvate per {dataset_name}")

def plot_regression_metrics(metrics, epochs, dataset_name):
    train_mse = metrics["train_loss"]
    val_mse = metrics["val_loss"]
    epoch_range = range(1, epochs + 1)  # 1-based index for epochs

    # Plot della Loss (MEE)
    plt.figure(figsize=(10, 5))
    #plt.plot(epoch_range, mean_train_loss, label="Train Loss")
    #plt.plot(epoch_range, mean_val_loss, label="Validation Loss", linestyle="--")
    plt.title(f"Loss Curve - {dataset_name} (Regression)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MEE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/loss_curve_{dataset_name}_regression.png", bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Learning curve (regression) salvata per {dataset_name}")


def plot_results_nn(task):
    # Unisce i risultati dei training e test sets
    merged_df = merge_results(results_tr, results_ts, task)
    print(f"Result TR: {results_tr}\nResult TS: {results_ts}")

    fig, ax = plt.subplots(figsize=(10, len(merged_df) * 0.6))  # Altezza dinamica in base al numero di righe
    ax.axis('tight')
    ax.axis('off')

    # Crea la tabella
    table = ax.table(cellText=merged_df.values, colLabels=merged_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(merged_df.columns))))

    # Salva il grafico come immagine
    plt.title(f"Risultati per Dataset - {task.capitalize()}")
    file_name = f"results/results_{task}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Tabella salvata: {file_name}")

def merge_results(results_train, results_test, task):
    # Converti le liste di dizionari in DataFrame
    df_train = pd.DataFrame(results_train)
    df_test = pd.DataFrame(results_test)

    # Rinomina le colonne per distinguere train, validation, test
    if task == "classification":
        df_train.rename(columns={"MSE (TR)": "MSE (TR)", "Accuracy (TR)": "Accuracy (TR)"}, inplace=True)
        df_test.rename(columns={"MSE (TS)": "MSE (TS)", "Accuracy (TS)": "Accuracy (TS)"}, inplace=True)
        column_order = ["Dataset", "MSE (TR)", "MSE (TS)", "Accuracy (TR)", "Accuracy (TS)"]
    elif task == "regression":
        df_train.rename(columns={"MEE (TR)": "MEE (TR)", "MEE (VL)": "MEE (VL)"}, inplace=True)
        df_test.rename(columns={"MEE (TS)": "MEE (TS)"}, inplace=True)
        column_order = ["Dataset", "MEE (TR)", "MEE (VL)", "MEE (TS)"]

    else:
        raise ValueError(f"Task '{task}' non riconosciuto. Usa 'classification' o 'regression'.")

    merged_df = pd.merge(df_train, df_test, on="Dataset", how="outer")

    # Ordina i risultati per Dataset e arrotonda i valori
    merged_df.sort_values(by="Dataset", inplace=True)
    merged_df = merged_df.round(4)

    # Mantieni solo le colonne disponibili
    available_columns = [col for col in column_order if col in merged_df.columns]
    merged_df = merged_df[available_columns]

    return merged_df

def predict_cup(model,x_ts):
    model.eval()

    with torch.no_grad():
        y_pred = model(x_ts)
    y_pred = y_pred.cpu().numpy()

    df_predictions = pd.DataFrame(y_pred, columns=["X", "Y", "Z"])
    file_path = os.path.join("results", "_ml_cup-24-ts.csv")
    df_predictions.to_csv(file_path, index=False)

def nn_execution(x_tr, y_tr, x_ts, y_ts = None, dataset = None, task = None):
    input_size = x_tr.shape[1]
    output_size = y_tr.shape[1]

    if task == "classification":
        best_model, best_params, metrics = optimize_hyperparameters(x_tr, y_tr, input_size, output_size, param_bounds, epochs, k, device, task)
        plot_classification_metrics(metrics, epochs, dataset)
        train_results = evaluation_tr(best_model, x_tr, y_tr, dataset , device, task) #mse (tr/ts) , accuracy (tr/ts)
        test_results = evaluation_ts(best_model, x_ts, y_ts, dataset, device, task)

        results_tr.append(train_results)
        results_ts.append(test_results)

    elif task == "regression":
        x_tr, x_its, y_tr,y_its = train_test_split(x_tr, y_tr, test_size=0.2, random_state=42)
        best_model, best_params, metrics = optimize_hyperparameters(x_tr, y_tr, input_size, output_size, param_bounds, epochs, k, device, task)
        #plot_regression_metrics(metrics, epochs, dataset,)
        train_results = evaluation_tr(best_model, x_tr, y_tr, dataset, device, task)  #mee (tr/vl/ts)
        test_results = evaluation_ts(best_model, x_its, y_its, dataset, device, task)

        results_tr.append(train_results)
        results_ts.append(test_results)

        #plot_learning_curve(avg_metrics, epochs, dataset)
        predict_cup(best_model, x_ts)


"""
def train_fold(model, x_train, y_train, x_val, y_val, epochs, lr, device, weight_decay, task):
    criterion = nn.MSELoss()  # MSE Loss per entrambi i task
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
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

        if task == "classification":
            train_preds = (outputs > 0.5).cpu().numpy()
            train_acc_history.append(accuracy_score(y_train.cpu(), train_preds))
        else:
            train_acc_history.append(np.nan)

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            val_loss_history.append(val_loss.item())

            if task == "classification":
                val_preds = (val_outputs > 0.5).cpu().numpy()
                val_acc_history.append(accuracy_score(y_val.cpu(), val_preds))
            else:
                val_acc_history.append(np.nan)

    if task == "classification":
        return {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "train_accuracy": train_acc_history,
            "val_accuracy": val_acc_history,
        }
    else:
        return {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
        }


def k_fold_cross_validation(x, y, input_size, output_size, hidden_size, epochs, lr, k, device, weight_decay, task):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    classification_metrics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    regression_metrics = {"train_loss": [], "val_loss": []}
    model_weights = []

    for train_idx, val_idx in kf.split(x, y):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Net(input_size, hidden_size, output_size, task).to(device)

        metrics = train_fold(model, x_train, y_train, x_val, y_val, epochs, lr, device, weight_decay, task)
        model_weights.append(model.state_dict())  # Salva i pesi del modello

        if task == "classification":
            classification_metrics["train_loss"].append(metrics["train_loss"])
            classification_metrics["val_loss"].append(metrics["val_loss"])
            classification_metrics["train_accuracy"].append(metrics["train_accuracy"])
            classification_metrics["val_accuracy"].append(metrics["val_accuracy"])
        else:
            regression_metrics["train_loss"].append(metrics["train_loss"])
            regression_metrics["val_loss"].append(metrics["val_loss"])

    averaged_weights = {k: sum(w[k] for w in model_weights) / len(model_weights) for k in model_weights[0]}

    if task == "classification":
        return classification_metrics, averaged_weights
    else:
        return regression_metrics, averaged_weights


def optimize_hyperparameters(x, y, input_size, output_size, param_grid, epochs, k, device, task):
    best_score = np.inf if task == "regression" else 0
    best_params = None
    best_metrics = None
    averaged_weights = None

    for hidden_size in param_grid["hidden_size"]:
        for lr in param_grid["lr"]:
            for weight_decay in param_grid["weight_decay"]:
                print(f"Training with hidden_size={hidden_size}, lr={lr}, weight_decay={weight_decay}")
                metrics, weights = k_fold_cross_validation(
                    x, y, input_size, output_size, hidden_size, epochs, lr, k, device, weight_decay, task
                )

                if task == "classification":
                    val_score = np.mean(metrics["val_accuracy"])
                    if val_score > best_score:
                        best_score = val_score
                        best_params = {"hidden_size": hidden_size, "lr": lr, "weight_decay": weight_decay}
                        best_metrics = metrics
                        averaged_weights = weights
                else:
                    val_score = np.mean(metrics["val_loss"])
                    if val_score < best_score:
                        best_score = val_score
                        best_params = {"hidden_size": hidden_size, "lr": lr, "weight_decay": weight_decay}
                        best_metrics = metrics
                        averaged_weights = weights

    best_model = Net(input_size, best_params["hidden_size"], output_size, task).to(device)
    best_model.load_state_dict(averaged_weights)

    print(f"Migliori parametri trovati: {best_params}")
    return best_model, best_params, best_metrics
"""





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

results_tr = []
results_ts = []

param_bounds = {
    "hidden_size": (16, 100),
    "lr": (0.0001, 0.01),
    "weight_decay": (0.0001, 0.01),
}
epochs = 500
k = 5
device = "cpu"

# PyTorch Model for Classification
class PyTorchNN(nn.Module):
    def init(self, input_size, hidden_size):
        super(PyTorchNN, self).init()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size,1)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)

        return x

def train_model(model, X_train, y_train, X_val, y_val, epochs, lr, device, weight_decay):
    criterion = nn.MSELoss()  # Solo per la classe positiva

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9 ,weight_decay=weight_decay,nesterov=True)
    model = model.to(device)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())
        train_preds = (outputs.sigmoid() > 0.5).cpu().numpy()
        train_acc_history.append(accuracy_score(y_train, train_preds))

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
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

    model.eval()
    with torch.no_grad():
        # Predizioni continue per il training set
        train_outputs = model(X_train).sigmoid()
        # Predizioni con la soglia ottimale
        train_preds = (train_outputs > 0.5).float()
        # Calcolo del train_loss
        train_loss = mean_squared_error(train_outputs, y_train)


        # Predizioni continue per il test set
        test_outputs = model(X_test).sigmoid()
        test_preds = (test_outputs > 0.5).float()
        # Calcolo del test_loss
        test_loss = mean_squared_error(test_outputs, y_test)

        # Accuratezza
        train_accuracy = accuracy_score(y_train.cpu(), train_preds.cpu())
        test_accuracy = accuracy_score(y_test.cpu(), test_preds.cpu())

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

def nn_execution(x_tr,y_tr,x_ts,y_ts,dataset):
    input_size = x_tr.shape[1]  # Rimosso la virgola
    output_size = y_tr.shape[1]  # Rimosso la virgola

     #_,best_p ,_ = optimize_hyperparameters(x_tr,y_tr,input_size,param_bounds,epochs,k,"cpu",dataset)
    
"""