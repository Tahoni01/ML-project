import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Funzione per l'addestramento del modello SVM
def train_svm(x_train, y_train, x_val, y_val, kernel, C, gamma):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(x_train_scaled, y_train)

    train_probs = model.predict_proba(x_train_scaled)
    val_probs = model.predict_proba(x_val_scaled)

    train_preds = model.predict(x_train_scaled)
    val_preds = model.predict(x_val_scaled)

    train_loss = log_loss(y_train, train_probs)
    val_loss = log_loss(y_val, val_probs)

    train_accuracy = accuracy_score(y_train, train_preds)
    val_accuracy = accuracy_score(y_val, val_preds)

    train_mse = np.mean((y_train - train_preds) ** 2)
    val_mse = np.mean((y_val - val_preds) ** 2)

    return model, train_loss, val_loss, train_accuracy, val_accuracy, train_mse, val_mse

# K-Fold Cross-Validation
def k_fold_cross_validation_svm(x, y, kernel, C, gamma, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_mse": [],
        "val_mse": []
    }

    for train_idx, val_idx in kf.split(x):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        _, train_loss, val_loss, train_accuracy, val_accuracy, train_mse, val_mse = train_svm(
            x_train, y_train, x_val, y_val, kernel, C, gamma
        )

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["train_mse"].append(train_mse)
        metrics["val_mse"].append(val_mse)

    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}

    return avg_metrics, metrics

# Grid Search con scikit-learn
def grid_search_cv_svm(x, y, param_grid, k=5):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = SVC(kernel="rbf", probability=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=k,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(x_scaled, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Migliori parametri: {best_params}")
    print(f"Accuratezza migliore: {best_score:.4f}")

    # Plot dei risultati del training
    results = grid_search.cv_results_
    plt.figure(figsize=(10, 6))
    plt.plot(results['param_C'], results['mean_test_score'], marker='o', label='Mean Test Accuracy')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('Mean Test Accuracy')
    plt.title('Grid Search Results')
    plt.legend()
    plt.grid(True)

    # Salva il grafico su file
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/grid_search_results.png")
    plt.close()

    return best_params, best_score

# Plot delle metriche
def plot_metrics(metrics, result_dir="results", prefix="svm"):
    os.makedirs(result_dir, exist_ok=True)
    epochs = range(1, len(metrics["train_loss"]) + 1)

    # Plot delle loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss", linestyle="--")
    plt.title(f"Loss Curves ({prefix})")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{result_dir}/{prefix}_loss_curves.png")
    plt.close()

    # Plot delle accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, metrics["val_accuracy"], label="Validation Accuracy", linestyle="--")
    plt.title(f"Accuracy Curves ({prefix})")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{result_dir}/{prefix}_accuracy_curves.png")
    plt.close()

    # Plot del MSE
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_mse"], label="Train MSE")
    plt.plot(epochs, metrics["val_mse"], label="Validation MSE", linestyle="--")
    plt.title(f"MSE Curves ({prefix})")
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.savefig(f"{result_dir}/{prefix}_mse_curves.png")
    plt.close()