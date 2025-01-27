from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

results_tr = []
results_ts = []

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

    return grid_search.best_estimator_

def model_creation(p):
    params = p.get_params()  # Ottieni i parametri dal modello SVC ottimizzato
    model = SVC(C=params["C"], kernel=params["kernel"], degree=params["degree"], gamma=params["gamma"])
    return model

def train_model(model,x,y,database):
    x_tr,x_vl,y_tr,y_vl = train_test_split(x,y,test_size=0.3, random_state=42)
    model.fit(x_tr,y_tr)

    y_pred = model.predict(x_vl)
    accuracy = accuracy_score(y_vl, y_pred)
    mse = mean_squared_error(y_vl, y_pred)

    results = {
        "Dataset": database,
        "MSE": mse,
        "Accuracy": accuracy
    }

    plot_learning_curve(model, x, y, database, "train")
    return results

def evaluation(model, x_ts, y_ts, database):
    y_pred = model.predict(x_ts)
    accuracy = accuracy_score(y_ts, y_pred)
    mse = mean_squared_error(y_ts, y_pred)

    results={
        "Dataset": database,
        "MSE": mse,
        "Accuracy": accuracy
    }

    plot_learning_curve(model, x_ts, y_ts, database, "evaluation")

    return results

def plot_learning_curve(model, x, y, database, task):
    train_sizes, train_scores, test_scores = learning_curve(
        model, x, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    # Convertire da negativo a positivo
    train_loss_mean = -np.mean(train_scores, axis=1)
    test_loss_mean = -np.mean(test_scores, axis=1)

    mean_train_loss = -np.mean(train_scores)
    mean_test_loss = -np.mean(test_scores)


    """
    print(f"TR loss: {mean_train_loss}")
    print(f"TS loss: {mean_test_loss}")
    """

    # Plot della curva
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_loss_mean, label="Loss (TR)", color="blue")
    plt.plot(train_sizes, test_loss_mean, label="Loss (VL)", color="orange")
    plt.title(f"Learning Curve - {task} - {database}")
    plt.xlabel("Training Size")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid()

    os.makedirs("results", exist_ok=True)  # Crea la directory se non esiste
    file_name = f"results/learning_curve_{task}_{database}.png"
    plt.savefig(file_name)
    plt.close()  # Chiude il grafico

def plot_results():
    merged_df = merge_results(results_tr,results_ts)
    fig, ax = plt.subplots(figsize=(10, len(merged_df) * 0.6))  # Altezza dinamica in base al numero di righe
    ax.axis('tight')
    ax.axis('off')

    # Crea la tabella
    table = ax.table(cellText=merged_df.values, colLabels=merged_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(merged_df.columns))))

    # Salva il grafico come immagine
    plt.title("Risultati per Dataset")
    file_name = f"results/results.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Tabella salvata")

def merge_results(results_train, results_test):
    """Unisce i risultati di train e test in una tabella unica con colonne ordinate."""
    # Converti le liste di dizionari in DataFrame
    df_train = pd.DataFrame(results_train)
    df_test = pd.DataFrame(results_test)

    # Rinomina le colonne di train e test per differenziarle
    df_train.rename(columns={"MSE": "MSE (TR)", "Accuracy": "Accuracy (TR)"}, inplace=True)
    df_test.rename(columns={"MSE": "MSE (TS)", "Accuracy": "Accuracy (TS)"}, inplace=True)

    # Unisci i due DataFrame sulla colonna "Dataset"
    merged_df = pd.merge(df_train, df_test, on="Dataset", how="outer")

    # Ordina i risultati per Dataset
    merged_df.sort_values(by="Dataset", inplace=True)

    # Approssima i valori a 10^-4
    merged_df = merged_df.round(4)

    # Riordina le colonne nel formato richiesto
    column_order = ["Dataset", "MSE (TR)", "MSE (TS)", "Accuracy (TR)", "Accuracy (TS)"]
    merged_df = merged_df[column_order]

    return merged_df

def svm_execution(x_tr, y_tr, x_ts, y_ts, database):
    p = optimize(x_tr,y_tr)
    model = model_creation(p)
    r_tr = train_model(model,x_tr,y_tr, database)
    r_ts = evaluation(model, x_ts, y_ts, database)

    results_tr.append(r_tr)
    results_ts.append(r_ts)

