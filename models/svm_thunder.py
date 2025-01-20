from thundersvm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
import numpy as np


def train_single_fold_thunder(X_train, y_train, X_val, y_val, kernel='rbf', C=1.0, gamma=None):
    """
    Addestra ThunderSVM su un singolo fold e calcola le metriche.

    Args:
        X_train (np.ndarray): Dati di training.
        y_train (np.ndarray): Target di training.
        X_val (np.ndarray): Dati di validazione.
        y_val (np.ndarray): Target di validazione.
        kernel (str): Tipo di kernel (es. 'linear', 'rbf', 'poly', 'sigmoid').
        C (float): Parametro di regolarizzazione.
        gamma (str): Coefficiente gamma per i kernel non lineari.

    Returns:
        dict: Metriche di accuratezza e perdita per il fold.
    """
    if gamma is None or isinstance(gamma, str):
        gamma = 1 / (X_train.shape[1] * np.var(X_train))  # Simile a 'scale' in scikit-learn

    # Inizializza il modello ThunderSVM
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

    # Addestra il modello
    model.fit(X_train, y_train)

    # Verifica se `predict_proba` è disponibile
    if hasattr(model, "predict_proba"):
        # Calcola le probabilità se possibile
        y_train_proba = model.predict_proba(X_train)
        y_val_proba = model.predict_proba(X_val)
        train_loss = log_loss(y_train, y_train_proba)
        val_loss = log_loss(y_val, y_val_proba)
    else:
        # Usa la distanza dai margini come sostituto delle probabilità
        y_train_decision = model.decision_function(X_train)
        y_val_decision = model.decision_function(X_val)
        train_loss = log_loss(y_train, 1 / (1 + np.exp(-y_train_decision)))
        val_loss = log_loss(y_val, 1 / (1 + np.exp(-y_val_decision)))

    # Accuratezza sui dati di validazione
    val_accuracy = accuracy_score(y_val, model.predict(X_val))

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }



def k_fold_cross_validation_thunder(X, y, k=5, kernel='rbf', C=1.0, gamma='scale'):
    """
    Esegue la K-Fold Cross-Validation con ThunderSVM e raccoglie i dati per la curva di apprendimento.

    Args:
        X (np.ndarray): Dati di input.
        y (np.ndarray): Target.
        k (int): Numero di fold.
        kernel (str): Tipo di kernel (es. 'linear', 'rbf', 'poly', 'sigmoid').
        C (float): Parametro di regolarizzazione.
        gamma (str): Coefficiente gamma per i kernel non lineari.

    Returns:
        list: Lista di metriche per ogni fold.
        dict: Metriche globali (media su tutti i fold).
        dict: Dati per la curva di apprendimento.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    train_sizes = []
    train_losses = []
    val_losses = []

    for train_index, val_index in kf.split(X):
        # Suddividi i dati
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Aggiungi la dimensione del training set
        train_sizes.append(len(X_train))

        # Addestra sul singolo fold
        metrics = train_single_fold_thunder(X_train, y_train, X_val, y_val, kernel, C, gamma)
        fold_metrics.append(metrics)

        # Salva le metriche di perdita
        train_losses.append(metrics["train_loss"])
        val_losses.append(metrics["val_loss"])

    # Calcola la media delle metriche
    overall_metrics = {
        "val_accuracy": np.mean([fold["val_accuracy"] for fold in fold_metrics]),
        "val_loss": np.mean([fold["val_loss"] for fold in fold_metrics])
    }

    # Dati per la curva di apprendimento
    learning_data = {
        "train_sizes": train_sizes,
        "train_loss": train_losses,
        "val_loss": val_losses
    }

    return fold_metrics, overall_metrics, learning_data
