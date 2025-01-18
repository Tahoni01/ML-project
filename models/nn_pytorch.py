"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# Classe della rete neurale
class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_size):

        Inizializza la rete neurale con un input layer, hidden layer e output layer.

        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Primo strato
        self.relu = nn.ReLU()  # Funzione di attivazione ReLU
        self.fc2 = nn.Linear(hidden_size, 1)  # Strato di output
        self.sigmoid = nn.Sigmoid()  # Funzione di attivazione Sigmoid

    def forward(self, x):

        Esegue il forward pass attraverso la rete neurale.

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Funzione per il training su un singolo fold
def train_single_fold(X_train, y_train, input_size, hidden_size, epochs=50, batch_size=32, lr=0.001):

    Addestra la rete neurale su un singolo fold.

    # Converti i dati in tensori
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Creare il DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Inizializza il modello
    model = PyTorchNN(input_size, hidden_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Addestramento
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    return model


# Funzione per la K-fold cross-validation
def k_fold_cross_validation(X, y, input_size, hidden_size, k=5, epochs=50, batch_size=32, lr=0.001):
    Esegue una validazione K-fold cross-validation.

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1} ---")

        # Suddivisione dei dati
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Addestramento del modello
        model = train_single_fold(X_train, y_train, input_size, hidden_size, epochs, batch_size, lr)

        # Valutazione sul fold di validazione
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

            outputs = model(X_val_tensor)
            predictions = (outputs >= 0.5).float()
            accuracy = accuracy_score(y_val_tensor.numpy(), predictions.numpy())
            fold_accuracies.append(accuracy)
            print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")

    return fold_accuracies
"""




"""
class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        Inizializza la rete neurale con un input layer, hidden layer e output layer.
    
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Primo strato
        self.relu = nn.ReLU()                         # Funzione di attivazione ReLU
        self.fc2 = nn.Linear(hidden_size, 1)          # Strato di output (1 neurone)
        self.sigmoid = nn.Sigmoid()                   # Funzione di attivazione Sigmoid

    def forward(self, x):
        
        Esegue il forward pass attraverso la rete neurale.

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_pytorch_nn(X_train, y_train, input_size, hidden_size, epochs=100, batch_size=32, lr=0.001):
    
    Addestra una rete neurale con PyTorch.

    Args:
        X_train (np.ndarray): Attributi di training.
        y_train (np.ndarray): Target di training.
        input_size (int): Dimensione dello strato di input.
        hidden_size (int): Numero di neuroni nello strato nascosto.
        epochs (int): Numero di epoche di allenamento.
        batch_size (int): Dimensione del batch.
        lr (float): Learning rate.

    Returns:
        model (PyTorchNN): Modello allenato.
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Convertire i dati in tensori PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Creare un DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Inizializzare il modello, la funzione di perdita e l'ottimizzatore
    model = PyTorchNN(input_size, hidden_size)
    criterion = nn.BCELoss()  # Perdita per classificazione binaria
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Variabili per il monitoraggio
    best_loss = float('inf')  # Inizializza best_loss a un valore molto alto
    best_model = None         # Salva il miglior modello durante il training

    # Addestramento
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass e aggiornamento dei pesi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Controllo per Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model
        else:
            print("Nessun miglioramento, interrompo il training.")
            break

    return best_model

def evaluate_pytorch_nn(model, X_test, y_test):
    
    Valuta il modello su un test set.

    Args:
        model (PyTorchNN): Modello allenato.
        X_test (np.ndarray): Attributi di test.
        y_test (np.ndarray): Target di test.

    Returns:
        accuracy (float): Accuratezza del modello.
    
    # Convertire i dati di test in tensori PyTorch
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Mettere il modello in modalità eval
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs >= 0.5).float()  # Classifica come 0 o 1
        accuracy = (predictions.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Classe della rete neurale
class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Inizializza la rete neurale con un input layer, hidden layer e output layer.
        Args:
            input_size (int): Numero di feature in input.
            hidden_size (int): Numero di neuroni nello strato nascosto.
        """
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Primo strato (Input -> Hidden)
        self.relu = nn.ReLU()  # Funzione di attivazione ReLU
        self.fc2 = nn.Linear(hidden_size, 1)  # Strato di output (Hidden -> Output)
        self.sigmoid = nn.Sigmoid()  # Funzione di attivazione Sigmoid

    def forward(self, x):
        """
        Esegue il forward pass attraverso la rete neurale.
        Args:
            x (torch.Tensor): Input.
        Returns:
            torch.Tensor: Output della rete.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Funzione per il training su un singolo fold
def train_single_fold(X_train, y_train, input_size, hidden_size, epochs=50, batch_size=32, lr=0.001):
    """
    Addestra la rete neurale su un singolo fold.
    Args:
        X_train (np.ndarray): Dati di input per il training.
        y_train (np.ndarray): Target per il training.
        input_size (int): Dimensione dello strato di input.
        hidden_size (int): Numero di neuroni nello strato nascosto.
        epochs (int): Numero di epoche di training.
        batch_size (int): Dimensione del batch.
        lr (float): Learning rate.
    Returns:
        PyTorchNN: Modello addestrato.
    """
    # Converti i dati in tensori PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Creare un DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Inizializza il modello
    model = PyTorchNN(input_size, hidden_size)
    criterion = nn.BCELoss()  # Perdita per classificazione binaria
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Addestramento
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass e aggiornamento dei pesi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    return model


# Funzione per valutare il modello
def evaluate_neural_network(model, X_test, y_test):
    """
    Valuta il modello su un test set.
    Args:
        model (PyTorchNN): Modello allenato.
        X_test (np.ndarray): Dati di input per il test.
        y_test (np.ndarray): Target per il test.
    Returns:
        float: Accuratezza del modello.
    """
    # Converti i dati in tensori PyTorch
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Modalità eval
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs >= 0.5).float()  # Soglia per classificazione binaria
        accuracy = (predictions.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Funzione per la K-fold cross-validation
def k_fold_cross_validation(X, y, input_size, hidden_size, k=5, epochs=50, batch_size=32, lr=0.001):
    """
    Esegue una validazione K-fold cross-validation.
    Args:
        X (np.ndarray): Dati di input.
        y (np.ndarray): Target binari.
        input_size (int): Dimensione dell'input.
        hidden_size (int): Numero di neuroni nello strato nascosto.
        k (int): Numero di fold.
        epochs (int): Numero di epoche.
        batch_size (int): Dimensione del batch.
        lr (float): Learning rate.
    Returns:
        list: Lista delle accuratezze per ogni fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        # Suddivisione dei dati
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Addestramento del modello
        model = train_single_fold(X_train, y_train, input_size, hidden_size, epochs, batch_size, lr)

        # Valutazione sul fold di validazione
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

            outputs = model(X_val_tensor)
            predictions = (outputs >= 0.5).float()
            accuracy = accuracy_score(y_val_tensor.numpy(), predictions.numpy())
            fold_accuracies.append(accuracy)
            print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")

    return fold_accuracies


# Funzione per il training e valutazione su tutti i dataset MONK
def train_and_evaluate_all(datasets, input_size, hidden_size, epochs=50, batch_size=32, lr=0.001):
    """
    Addestra e valuta il modello su tutti i dataset MONK.
    Args:
        datasets (dict): Dataset preprocessati.
        input_size (int): Dimensione dell'input.
        hidden_size (int): Numero di neuroni nello strato nascosto.
        epochs (int): Numero di epoche.
        batch_size (int): Dimensione del batch.
        lr (float): Learning rate.
    Returns:
        dict: Accuratezze per ogni dataset.
    """
    results = {}
    for monk_name, data in datasets.items():
        print(f"\n--- Addestramento e valutazione per {monk_name} ---")

        # Estrai i dati
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        # Addestramento
        model = train_single_fold(X_train, y_train, input_size, hidden_size, epochs, batch_size, lr)

        # Valutazione
        accuracy = evaluate_neural_network(model, X_test, y_test)
        results[monk_name] = accuracy

        print(f"Accuratezza per {monk_name}: {accuracy * 100:.2f}%\n")

    return results
