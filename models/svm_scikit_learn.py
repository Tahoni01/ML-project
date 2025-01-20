from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,accuracy_score
import numpy as np

def train_single_fold(X_train, y_train, X_val, y_val, kernel='linear', C=1.0):
    """
    Addestra un modello SVM su un singolo fold.

    Args:
        X_train (np.ndarray): Dati di input per il training.
        y_train (np.ndarray): Target per il training.
        X_val (np.ndarray): Dati di input per la validazione.
        y_val (np.ndarray): Target per la validazione.
        kernel (str): Tipo di kernel per SVM (es. 'linear', 'rbf').
        C (float): Parametro di regolarizzazione.

    Returns:
        dict: Metriche di valutazione del modello.
    """
    model = SVC(kernel=kernel, C=C, probability=True)
    model.fit(X_train, y_train)

    # Predizioni su training e validation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Predizioni probabilistiche per calcolare la log-loss
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)

    # Calcola metriche
    metrics = {
        "train_loss": log_loss(y_train, y_train_proba),
        "val_loss": log_loss(y_val, y_val_proba),
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "val_accuracy": accuracy_score(y_val, y_val_pred)
    }

    return metrics

def k_fold_cross_validation(X, y, kernel='linear', C=1.0, k=5):
    """
    Esegue una K-Fold Cross-Validation e raccoglie i dati per la curva di apprendimento.

    Returns:
        fold_histories: Storico delle perdite per ogni fold.
        overall_history: Risultati medi sui fold.
        learning_data: Dati per il grafico di apprendimento.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_histories = []
    train_sizes = []
    train_loss = []
    val_loss = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Addestra il modello
        model = SVC(kernel=kernel, C=C, probability=True)
        model.fit(X_train, y_train)

        # Calcola la perdita
        train_proba = model.predict_proba(X_train)
        val_proba = model.predict_proba(X_val)

        fold_train_loss = log_loss(y_train, train_proba)
        fold_val_loss = log_loss(y_val, val_proba)

        fold_histories.append({
            "train_loss": fold_train_loss,
            "val_loss": fold_val_loss
        })

        # Raccogli i dati per la curva di apprendimento
        train_sizes.append(len(X_train))
        train_loss.append(fold_train_loss)
        val_loss.append(fold_val_loss)

    overall_history = {
        "train_loss": np.mean([h["train_loss"] for h in fold_histories]),
        "val_loss": np.mean([h["val_loss"] for h in fold_histories]),
    }

    learning_data = {
        "train_sizes": train_sizes,
        "train_loss": train_loss,
        "val_loss": val_loss
    }

    return fold_histories, overall_history, learning_data

