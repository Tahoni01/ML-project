import jax
import jax.numpy as jnp
from jax import grad, jit
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# Definizione del modello
def initialize_params(input_size, hidden_size, output_size):
    """
    Inizializza i pesi e i bias del modello.
    """
    key = jax.random.PRNGKey(0)
    w1 = jax.random.normal(key, shape=(input_size, hidden_size))
    b1 = jnp.zeros(hidden_size)
    w2 = jax.random.normal(key, shape=(hidden_size, output_size))
    b2 = jnp.zeros(output_size)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def forward(params, x):
    """
    Propagazione in avanti.
    """
    z1 = jnp.dot(x, params["w1"]) + params["b1"]
    h1 = jax.nn.relu(z1)
    z2 = jnp.dot(h1, params["w2"]) + params["b2"]
    return jax.nn.sigmoid(z2)


def binary_cross_entropy_loss(params, x, y):
    """
    Calcola la perdita Binary Cross-Entropy.
    """
    preds = forward(params, x)
    preds = jnp.clip(preds, 1e-7, 1 - 1e-7)  # Evita problemi numerici
    return -jnp.mean(y * jnp.log(preds) + (1 - y) * jnp.log(1 - preds))


def accuracy(params, x, y):
    """
    Calcola l'accuratezza.
    """
    preds = forward(params, x)
    return jnp.mean((preds >= 0.5) == y)


# Funzione per addestrare su un singolo fold
def train_single_fold(X_train, y_train, X_val, y_val, input_size, hidden_size, epochs=50, lr=0.001):
    """
    Addestra il modello JAX su un singolo fold.
    """
    params = initialize_params(input_size, hidden_size, 1)
    grad_loss = grad(binary_cross_entropy_loss)  # Calcola il gradiente della perdita
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for epoch in range(epochs):
        grads = grad_loss(params, X_train, y_train)
        # Aggiorna i pesi con gradient descent
        params = jax.tree_map(lambda p, g: p - lr * g, params, grads)

        # Calcola la perdita e l'accuratezza per training e validation
        train_loss = binary_cross_entropy_loss(params, X_train, y_train)
        train_acc = accuracy(params, X_train, y_train)
        val_loss = binary_cross_entropy_loss(params, X_val, y_val)
        val_acc = accuracy(params, X_val, y_val)

        # Salva i risultati
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

    return params, history


# Funzione per K-Fold Cross-Validation
def k_fold_cross_validation(X, y, input_size, hidden_size, k=5, epochs=50, lr=0.001):
    """
    Esegue la K-Fold Cross-Validation utilizzando JAX.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{k} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        _, history = train_single_fold(
            X_train, y_train, X_val, y_val, input_size, hidden_size, epochs, lr
        )
        fold_histories.append(history)

    # Converti le metriche finali in array JAX
    val_accuracies = jnp.array([history["val_accuracy"][-1] for history in fold_histories])
    val_losses = jnp.array([history["val_loss"][-1] for history in fold_histories])

    # Calcola metriche medie
    avg_metrics = {
        "accuracy": jnp.mean(val_accuracies),
        "loss": jnp.mean(val_losses),
    }

    return fold_histories, avg_metrics

