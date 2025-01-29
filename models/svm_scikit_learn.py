from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

results_tr = []
results_ts = []

param_grid_svc = {
        'kernel': ['linear','rbf', 'poly', 'sigmoid'],  # Testiamo kernel rbf e poly
        'C': [0.1, 1, 5, 10, 50, 100],  # Parametro di regolarizzazione
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01],  # Gamma automatico
        'degree': [2, 3, 4, 5]}

param_grid_svr = {
        'estimator__kernel': ['linear','rbf', 'poly', 'sigmoid'],  # Testiamo kernel rbf e poly
        'estimator__C': [0.1, 1, 5, 10, 50, 100],  # Parametro di regolarizzazione
        'estimator__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01],  # Gamma automatico
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0]}

def mee(y_true, y_pred):
    errors = y_true - y_pred
    return np.linalg.norm(errors, axis=1).mean()


mee_scorer = make_scorer(mee, greater_is_better=False)  # Negativo per minimizzare


def optimize(x_tr,y_tr,task):

    if task == "classification":
        svc = SVC(class_weight='balanced')
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)  # Stratified K-Fold
        grid_search = GridSearchCV(estimator=svc, param_grid=param_grid_svc,cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1, error_score='raise')
    else:
        svr = MultiOutputRegressor(SVR())
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)  # K-Fold
        grid_search = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=kfold,scoring=mee_scorer, n_jobs=-1)

    grid_search.fit(x_tr, y_tr)

    return grid_search.best_params_

def model_creation(best_params, task):
    if task == "classification":
        # Crea un modello SVC con i parametri ottimizzati
        model = SVC(C=best_params["C"], kernel=best_params["kernel"], degree=best_params["degree"], gamma=best_params["gamma"])
    else:
        # Crea un modello SVR o MultiOutputRegressor(SVR) con i parametri ottimizzati
        svr_params = {
            "C": best_params["estimator__C"],
            "kernel": best_params["estimator__kernel"],
            "epsilon": best_params["estimator__epsilon"],
            "gamma": best_params["estimator__gamma"]
        }
        svr = SVR(**svr_params)  # Costruisci il modello SVR
        model = MultiOutputRegressor(svr)  # Wrappa il modello SVR in MultiOutputRegressor per regressione multivariata

    return model

def evaluation_tr(model, x, y, dataset, task):
    x_tr,x_vl,y_tr,y_vl = train_test_split(x,y,test_size=0.2, random_state=42)

    model.fit(x_tr,y_tr)

    # Predizioni per il training set (TR), il validation set (VL), e il test set (TS)
    y_pred_tr = model.predict(x_tr)  # Predizione su training set
    y_pred_vl = model.predict(x_vl)  # Predizione su validation set

    if task == "classification":
        accuracy = accuracy_score(y_vl, y_pred_vl)
        mse = mean_squared_error(y_vl, y_pred_vl)
        results = {
            "Dataset": dataset,
            "MSE": mse,
            "accuracy": accuracy
        }
        plot_loss_curve(model, x, y, dataset, "classification","neg_mean_squared_error", "TR")
    else:
        mee_tr = mee(y_tr, y_pred_tr)
        mee_vl = mee(y_vl, y_pred_vl)
        results = {
            "Dataset": dataset,
            "MEE (TR)": mee_tr,
            "MEE (VL)": mee_vl
        }
        plot_loss_curve(model, x, y, dataset, "regression", mee_scorer, "TR")

    return results

def evaluation_ts(model, x_ts, y_ts, database, task):
    y_pred = model.predict(x_ts)

    if task == "classification":
        mse = mean_squared_error(y_ts, y_pred)
        accuracy = accuracy_score(y_ts, y_pred)
        results={
            "Dataset": database,
            "MSE": mse,
            "accuracy":accuracy}
        plot_loss_curve(model, x_ts, y_ts, database ,"classification", "neg_mean_squared_error", "TS")
    else:
        mee_score = mee(y_ts,y_pred)
        results={
            "Dataset": database,
            "MEE (TS)": mee_score,
        }
        plot_loss_curve(model, x_ts, y_ts, database, "regression", mee_scorer, "TS")

    return results


def plot_loss_curve(model, x, y, database, task, scoring_metric="neg_mean_squared_error", section = ""):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Calcolo della learning curve per la loss
    train_sizes, train_scores, test_scores = learning_curve(
        model, x, y, cv=kfold, scoring=scoring_metric,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_loss_mean = -np.mean(train_scores, axis=1)  # Converti in valori positivi
    test_loss_mean = -np.mean(test_scores, axis=1)

    # Creazione del grafico
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_loss_mean, label="Loss (TR)", color="blue")
    plt.plot(train_sizes, test_loss_mean, label="Loss (VL)", color="orange", linestyle="--")
    plt.title(f"Learning Curve - Loss ({task.capitalize()}) - {database}")
    plt.xlabel("Training Size")
    if task == "regression": plt.ylabel("Loss (MEE)")
    elif task == "classification": plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid()

    # Salvataggio del grafico
    os.makedirs("results", exist_ok=True)
    file_name = f"results/{section}_loss_curve_{task}_{database}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Loss curve salvata: {file_name}")

def plot_results_svm(task):
    merged_df = merge_results(results_tr, results_ts, task)
    fig, ax = plt.subplots(figsize=(10, len(merged_df) * 0.6))  # Altezza dinamica in base al numero di righe
    ax.axis('tight')
    ax.axis('off')

    # Crea la tabella
    table = ax.table(cellText=merged_df.values, colLabels=merged_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(merged_df.columns))))

    # Salva il grafico come immagine
    plt.title(f"Risultati per Dataset - {task.capitalize()}")
    file_name = f"results/results_{task}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Tabella salvata: {file_name}")

def merge_results(results_train, results_test, task):
    # Converti le liste di dizionari in DataFrame
    df_train = pd.DataFrame(results_train)
    df_test = pd.DataFrame(results_test)

    # Rinomina le colonne per distinguere train e test
    if task == "classification":
        df_train.rename(columns={"MSE": "MSE (TR)", "accuracy": "Accuracy (TR)"}, inplace=True)
        df_test.rename(columns={"MSE": "MSE (TS)", "accuracy": "Accuracy (TS)"}, inplace=True)
        column_order = ["Dataset", "MSE (TR)", "MSE (TS)", "Accuracy (TR)", "Accuracy (TS)"]
    elif task == "regression":
        df_train.rename(columns={"MEE (TR)": "MEE (TR)", "MEE (VL)": "MEE (VL)"}, inplace=True)
        df_test.rename(columns={"MEE (TS)": "MEE (TS)"}, inplace=True)
        column_order = ["Dataset", "MEE (TR)", "MEE (VL)", "MEE (TS)"]
    else:
        raise ValueError(f"Task '{task}' non riconosciuto. Usa 'classification' o 'regression'.")

    merged_df = pd.merge(df_train, df_test, on="Dataset", how="outer")

    # Ordina i risultati per Dataset e arrotonda i valori
    merged_df.sort_values(by="Dataset", inplace=True)
    merged_df = merged_df.round(4)

    # Mantieni solo le colonne disponibili
    available_columns = [col for col in column_order if col in merged_df.columns]
    merged_df = merged_df[available_columns]

    return merged_df

def predict_cup(model, x_ts):
    y_pred = model.predict(x_ts)  # Usa il modello addestrato per fare le previsioni
    df_predictions = pd.DataFrame(y_pred, columns=["X", "Y", "Z"])

    file_path = os.path.join("results", "team_ml_cup-24-ts.csv")
    df_predictions.to_csv(file_path, index=False)

    print(f"Predizioni salvate")


def svm_execution(x_train, y_train, x_test, y_test= None, dataset=None, task=None):
    if task == "regression":
        x_tr, x_ts, y_tr,y_ts = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        p = optimize(x_tr, y_tr, task)
        model = model_creation(p, task)
        r_tr = evaluation_tr(model, x_tr, y_tr, dataset, task)
        r_ts = evaluation_ts(model, x_ts, y_ts, dataset, task)
        results_tr.append(r_tr)
        results_ts.append(r_ts)
        predict_cup(model, x_test)
    elif task == "classification":
        p = optimize(x_train, y_train, task)
        model = model_creation(p, task)
        r_tr = evaluation_tr(model, x_train, y_train, dataset, task)
        r_ts = evaluation_ts(model, x_test, y_test, dataset, task)
        results_tr.append(r_tr)
        results_ts.append(r_ts)



