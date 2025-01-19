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
    Esegue la K-Fold Cross-Validation per SVM.

    Args:
        X (np.ndarray): Dati di input.
        y (np.ndarray): Target binari.
        kernel (str): Tipo di kernel per SVM.
        C (float): Parametro di regolarizzazione.
        k (int): Numero di fold.

    Returns:
        list: Lista delle metriche per ogni fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        metrics = train_single_fold(X_train, y_train, X_val, y_val, kernel, C)
        fold_metrics.append(metrics)

    # Calcola le medie delle metriche su tutti i fold
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0]
    }

    return fold_metrics, avg_metrics
