from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import os


def optimize(x_tr,y_tr):
    param_grid = {
        'kernel': ['linear','rbf', 'poly', 'sigmoid'],  # Testiamo kernel rbf e poly
        'C': [0.1, 1, 5, 10, 50, 100],  # Parametro di regolarizzazione
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01],  # Gamma automatico
        'degree': [2, 3, 4, 5]  # Gradi per il kernel polinomiale
    }
    svc = SVC(class_weight='balanced')
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Stratified K-Fold
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid,cv=stratified_kfold, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(x_tr, y_tr)
    print("Migliori parametri trovati:")
    print(grid_search.best_params_)
    return grid_search.best_estimator_

def model_creation(p):
    params = p.get_params()  # Ottieni i parametri dal modello SVC ottimizzato
    model = SVC(C=params["C"], kernel=params["kernel"], degree=params["degree"], gamma=params["gamma"])
    return model

def train_model(model,x,y):
    x_tr,x_vl,y_tr,y_vl = train_test_split(x,y,test_size=0.3, random_state=42)
    model.fit(x_tr,y_tr)

    plot_learning_curve(model, x, y)

    y_pred = model.predict(x_vl)
    accuracy = accuracy_score(y_vl, y_pred)
    mse = mean_squared_error(y_vl, y_pred)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    print(f"Training MSE: {mse:.4f}")

def evaluation(model, x_ts, y_ts):
    y_pred = model.predict(x_ts)
    accuracy = accuracy_score(y_ts, y_pred)
    mse = mean_squared_error(y_ts, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test MSE: {mse:.4f}")

def plot_learning_curve(model, x, y):
    """Genera e plotta la learning curve basata sul MSE."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, x, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    # Convertire da negativo a positivo
    train_loss_mean = -np.mean(train_scores, axis=1)
    test_loss_mean = -np.mean(test_scores, axis=1)

    mean_train_loss = -np.mean(train_scores)
    mean_test_loss = -np.mean(test_scores)

    print(f"TR loss: {mean_train_loss}")
    print(f"TS loss: {mean_test_loss}")

    # Plot della curva
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_loss_mean, label="Loss (TR)", color="blue")
    plt.plot(train_sizes, test_loss_mean, label="Loss (VL)", color="orange")
    plt.title(f"SVR Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid()

    os.makedirs("results", exist_ok=True)  # Crea la directory se non esiste
    plt.savefig("results/learning_curve.png")
    plt.close()  # Chiude il grafico

def svm_execution(x_tr, y_tr, x_ts, y_ts):
    p = optimize(x_tr,y_tr)
    model = model_creation(p)
    train_model(model,x_tr,y_tr)
    evaluation(model, x_ts, y_ts)