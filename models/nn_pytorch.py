import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

results_tr = []
results_ts = []

param_bounds = {
    'hidden_size': (2,4),
    'lr': (0.009, 0.01),
    'weight_decay': (0.0001, 0.001)
}
epochs = 500
k = 5
device = "cpu"

def mee(y_true, y_pred):
    errors = y_true - y_pred
    return torch.norm(errors.detach(), p=2, dim=1).mean()

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


def train_fold(model, x_train, y_train, x_val, y_val, epochs, lr, device, weight_decay, task):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=False)
    model = model.to(device)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    train_mee_history, val_mee_history = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())
        train_preds = (outputs > 0.5).cpu().numpy()
        if task == "classification": train_acc_history.append(accuracy_score(y_train, train_preds))
        else: train_mee_history.append(mee(y_train, outputs))


        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            val_loss_history.append(val_loss.item())
            val_preds = (val_outputs > 0.5).cpu().numpy()
            if task == "classification": val_acc_history.append(accuracy_score(y_val, val_preds))
            else: val_mee_history.append(mee(y_val, val_outputs))

    if task == "classification":
        return {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "train_accuracy": train_acc_history,
            "val_accuracy": val_acc_history}
    else:
        return {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "train_mee": train_mee_history,
            "val_mee": val_mee_history,
        }


def k_fold_cross_validation(x, y, input_size, output_size, hidden_size, epochs, lr, k, device, weight_decay, task):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_mse = []
    if task == "classification":
        all_metrics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    else:
        all_metrics = {"train_loss": [], "val_loss": [], "train_mee": [], "val_mee": []}

    model_weights = []

    for train_idx, val_idx in kf.split(x, y):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Net(input_size, hidden_size, output_size, task).to(device)

        metrics = train_fold(model, x_train, y_train, x_val, y_val, epochs, lr, device, weight_decay, task)
        model_weights.append(model.state_dict())

        val_mse.append(np.mean(metrics["val_loss"]))

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    avg_metrics = {
        key: np.nanmean([np.pad(m, (0, max(map(len, all_metrics[key])) - len(m)), constant_values=np.nan) for m in
                         all_metrics[key]], axis=0).tolist()
        for key in all_metrics
    }

    averaged_weights = {k: sum(w[k] for w in model_weights) / len(model_weights) for k in model_weights[0]}

    return np.mean(val_mse), avg_metrics, averaged_weights


def optimize_hyperparameters(x, y, input_size, output_size, param_bounds, epochs, k, device, task):
    best_score = float('inf')
    best_params = None
    avg_metrics = None
    averaged_weights = None

    for _ in range(20):
        hidden_size = random.randint(*param_bounds["hidden_size"])
        lr = random.uniform(*param_bounds["lr"])
        weight_decay = random.uniform(*param_bounds["weight_decay"])

        val_mse_mean, metrics, weights = k_fold_cross_validation(
            x, y, input_size, output_size, hidden_size, epochs, lr, k,device, weight_decay, task
        )

        if val_mse_mean < best_score:
            best_score = val_mse_mean
            best_params = {"hidden_size": hidden_size, "lr": lr, "weight_decay":weight_decay}
            avg_metrics = metrics
            averaged_weights = weights

    best_model = Net(input_size, best_params["hidden_size"], output_size, task).to(device)
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
            mee_tr = mee(y_tr.cpu().detach(), y_pred_tr.cpu().detach()).item()
            mee_vl = mee(y_vl.cpu().detach(), y_pred_vl.cpu().detach()).item()
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
            mee_ts = mee(y.cpu().detach(), y_pred_ts.cpu().detach()).item()
            results = {
                "Dataset": dataset_name,
                "MEE (TS)": mee_ts,
            }
    return results

def plot_classification_metrics(metrics, epochs, dataset_name):
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    train_accuracy = metrics["train_accuracy"]
    val_accuracy = metrics["val_accuracy"]

    epoch_range = np.arange(1, len(train_loss) + 1)

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
    train_mee = metrics["train_mee"]
    val_mee = metrics["val_mee"]
    epoch_range = range(1, epochs + 1)  # 1-based index for epochs

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_mee, label="Train Loss")
    plt.plot(epoch_range, val_mee, label="Validation Loss", linestyle="--")
    plt.title(f"Loss Curve - {dataset_name} (Regression)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MEE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/loss_curve_{dataset_name}_regression.png", bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Learning curve (regression) salvata per {dataset_name}")


def plot_results_nn(task):
    merged_df = merge_results(results_tr, results_ts, task)
    print(f"Result TR: {results_tr}\nResult TS: {results_ts}")

    fig, ax = plt.subplots(figsize=(10, len(merged_df) * 0.6))  # Altezza dinamica in base al numero di righe
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=merged_df.values, colLabels=merged_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(merged_df.columns))))

    plt.title(f"Risultati per Dataset - {task.capitalize()}")
    file_name = f"results/results_{task}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Tabella salvata: {file_name}")

def merge_results(results_train, results_test, task):
    df_train = pd.DataFrame(results_train)
    df_test = pd.DataFrame(results_test)

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

    merged_df.sort_values(by="Dataset", inplace=True)
    merged_df = merged_df.round(4)

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
        train_results = evaluation_tr(best_model, x_tr, y_tr, dataset , device, task)
        test_results = evaluation_ts(best_model, x_ts, y_ts, dataset, device, task)

        results_tr.append(train_results)
        results_ts.append(test_results)

    elif task == "regression":
        x_tr, x_its, y_tr,y_its = train_test_split(x_tr, y_tr, test_size=0.2, random_state=42)
        best_model, best_params, metrics = optimize_hyperparameters(x_tr, y_tr, input_size, output_size, param_bounds, epochs, k, device, task)
        plot_regression_metrics(metrics, epochs, dataset)
        train_results = evaluation_tr(best_model, x_tr, y_tr, dataset, device, task)
        test_results = evaluation_ts(best_model, x_its, y_its, dataset, device, task)

        results_tr.append(train_results)
        results_ts.append(test_results)

        predict_cup(best_model, x_ts)
