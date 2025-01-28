from preprocess import load_all_monks, load_cup_data
from models.svm_scikit_learn import svm_execution, plot_results_svm
from models.nn_pytorch import nn_execution, plot_results_nn
import os
from concurrent.futures import ThreadPoolExecutor

def main():
    print("Caricamento dei dati...")

    # Path ai dataset
    base_path_monk = "data/monk/"
    cup_train_path = "data/cup/ML-CUP24-TR.csv"
    cup_test_path = "data/cup/ML-CUP24-TS.csv"

    # Caricamento dei dataset MONK
    monk_datasets = load_all_monks(base_path_monk)
    # Caricamento dei dati CUP
    cup_data = load_cup_data(cup_train_path, cup_test_path)

    # Creazione della directory per i risultati
    result_dir = "results/"
    os.makedirs(result_dir, exist_ok=True)

    # Esecuzione parallela di analisi
    with ThreadPoolExecutor() as executor:
        futures = []

        # Analisi per i dataset MONK
        for dataset_name, data in monk_datasets.items():
            print(f"\n--- Analisi del dataset MONK: {dataset_name} ---")

            # Training e test set
            X_train, Y_train = data["train"]["torch"]
            X_test, Y_test = data["test"]["torch"]
            input_size = X_train.shape[1]
            """
            # Analisi SVM
            futures.append(
                executor.submit(svm_execution, X_train, Y_train, X_test, Y_test, dataset_name, "classification")
            )
            """

            # Analisi PyTorch
            futures.append(
                executor.submit(nn_execution, X_train, Y_train, X_test, Y_test, input_size, 1, dataset_name)
            )

        """
        # Analisi per il dataset CUP
        print("\n--- Analisi del dataset CUP ---")
        X_train_cup, Y_train_cup = cup_data["train"]["numpy"]
        X_test_cup, Y_test_cup = cup_data["test"]["numpy"]
        input_size_cup = X_train_cup.shape[1]

        # Analisi SVM per CUP
        futures.append(
            executor.submit(svm_execution, X_train_cup, Y_train_cup, X_train_cup, Y_train_cup, "CUP", "regression")
        )

        
        # Analisi PyTorch per CUP
        futures.append(
            executor.submit(nn_execution, X_train_cup, Y_train_cup, X_test_cup, Y_test_cup, input_size_cup, 3, "CUP")
        )
        """

        # Attendi il completamento di tutti i task
        for future in futures:
            future.result()

    # Genera i plot dei risultati
    plot_results_svm()
    # plot_results_nn()

    print("\nAnalisi completata per tutti i dataset!")\

if __name__ == "__main__":
    main()