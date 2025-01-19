"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Modelli di reti neurali
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class TwoHiddenLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TwoHiddenLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class DropoutNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(DropoutNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Funzione per il training su un singolo fold
def train_single_fold(X_train, y_train, model_class, device, input_size, hidden_size, epochs=50, batch_size=32, lr=0.001):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model_class(input_size, hidden_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return model

# Funzione per la K-Fold Cross-Validation
def k_fold_cross_validation(X, y, model_class, input_size, hidden_size, k=5, epochs=50, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = train_single_fold(
            X_train, y_train, model_class, device, input_size, hidden_size, epochs, batch_size, lr
        )

        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

            outputs = model(X_val_tensor)
            predictions = (outputs >= 0.5).float()

            accuracy = accuracy_score(y_val_tensor.cpu(), predictions.cpu())
            precision = precision_score(y_val_tensor.cpu(), predictions.cpu(), zero_division=0)
            recall = recall_score(y_val_tensor.cpu(), predictions.cpu(), zero_division=0)

            f1 = f1_score(y_val_tensor.cpu(), predictions.cpu())

            fold_metrics.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    metrics_mean = {metric: torch.tensor([fold[metric] for fold in fold_metrics]).mean().item()
                    for metric in ["accuracy", "precision", "recall", "f1"]}
    return metrics_mean
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class TwoHiddenLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TwoHiddenLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class DropoutNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(DropoutNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_single_fold(X_train, y_train, X_val, y_val, model_class, input_size, hidden_size, epochs=50, batch_size=32, lr=0.001):
    """
    Addestra un modello PyTorch su un singolo fold.
    """
    # Conversione dei dati in tensori
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Dataset e DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Modello, perdita e ottimizzatore
    model = model_class(input_size, hidden_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct_train += (predictions == targets).sum().item()
            total_train += targets.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Valutazione
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_tensor)
            val_loss = criterion(outputs_val, y_val_tensor).item()
            predictions_val = (outputs_val >= 0.5).float()
            val_accuracy = (predictions_val == y_val_tensor).sum().item() / y_val_tensor.size(0)

        # Aggiungi alle metriche
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    return model, history


def k_fold_cross_validation(X, y, model_class, input_size, hidden_size, k=5, epochs=50, batch_size=32, lr=0.001):
    """
    Esegue una K-Fold Cross-Validation.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{k} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        _, history = train_single_fold(
            X_train, y_train, X_val, y_val, model_class, input_size, hidden_size, epochs, batch_size, lr
        )
        fold_histories.append(history)

    return fold_histories
