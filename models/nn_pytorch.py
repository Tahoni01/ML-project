import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Inizializza la rete neurale con un input layer, hidden layer e output layer.
        """
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Primo strato
        self.relu = nn.ReLU()                         # Funzione di attivazione ReLU
        self.fc2 = nn.Linear(hidden_size, 1)          # Strato di output (1 neurone)
        self.sigmoid = nn.Sigmoid()                   # Funzione di attivazione Sigmoid

    def forward(self, x):
        """
        Esegue il forward pass attraverso la rete neurale.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_pytorch_nn(X_train, y_train, input_size, hidden_size, epochs=100, batch_size=32, lr=0.001):
    """
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
    """
    # Normalizza i target in 0 e 1
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())

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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    return model

def evaluate_pytorch_nn(model, X_test, y_test):
    """
    Valuta il modello su un test set.

    Args:
        model (PyTorchNN): Modello allenato.
        X_test (np.ndarray): Attributi di test.
        y_test (np.ndarray): Target di test.

    Returns:
        accuracy (float): Accuratezza del modello.
    """
    # Normalizza i target in 0 e 1
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

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








"""import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Preprocessing e Caricamento Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizzazione su media=0.5, std=0.5
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. Definizione del Modello
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 28x28 immagini, 128 neuroni
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Output: 10 classi

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten dell'immagine
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()

# 3. Funzione di Perdita e Ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass e Aggiornamento Pesi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 5. Valutazione
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

# 6. Esecuzione
train(model, train_loader, criterion, optimizer, epochs=5)
evaluate(model, test_loader)
"""