from preprocess import load_all_monks, load_monk_data
from models.nn_pytorch import optimize_hyperparameters,final_evaluation
from models.svm_scikit_learn import svm_execution, plot_results
import os
import numpy as np

def main():
    print("Caricamento dei dati...")
    base_path = "data/monk/"
    datasets = ["monks-1", "monks-2", "monks-3"]

    # Directory per salvare i risultati
    result_dir = "results/"
    os.makedirs(result_dir, exist_ok=True)

    # Parametri per le reti neurali
    nn_param_bounds = {
        "hidden_size": (16, 100),
        "lr": (0.005,0.5),
        "weight_decay": (0.0001, 0.01),
    }
    nn_epochs = 500
    nn_k = 5
    device = "cpu"

    # Itera su tutti i dataset Monk
    for dataset in datasets:
        print(f"\n--- Analisi del dataset {dataset} ---")

        # Carica il training set e il test set
        train_path = os.path.join(base_path, f"{dataset}.train")
        test_path = os.path.join(base_path, f"{dataset}.test")
        x, y = load_monk_data(train_path)
        z, w = load_monk_data(test_path)

        input_size = x.shape[1]

        """
        # Analisi con reti neurali
        print(f"-> Ottimizzazione NN per il dataset {dataset}")
        nn_model, nn_best_params, nn_best_val_accuracy = optimize_hyperparameters(
            x, y, input_size, nn_param_bounds, nn_epochs, nn_k, device, dataset_name=dataset
        )
        final_evaluation(nn_model, x, y, z, w, device, dataset_name=f"{dataset}_NN")
        """

        # Analisi con SVM
        print(f"-> Ottimizzazione SVM per il dataset {dataset}")
        svm_execution(x,y,z,w,dataset)
        plot_results()


    print("\nAnalisi completata per tutti i dataset Monk!")

if __name__ == "__main__":
    main()