from preprocess import load_all_monks, load_cup_data
from models.svm_scikit_learn import svm_execution, plot_results_svm
from models.nn_pytorch import nn_execution, plot_results_nn
import os

def main():
    print("Caricamento dei dati...")

    # Path ai dataset
    base_path_monk = "data/monk/"

    cup_train_path = "data/cup/ML-CUP24-TR.csv"
    cup_test_path = "data/cup/ML-CUP24-TS.csv"

    monk_datasets, encoders, scalers = load_all_monks(base_path_monk)
    cup_data = load_cup_data(cup_train_path, cup_test_path)

    result_dir = "results/"
    os.makedirs(result_dir, exist_ok=True)

    model_input = input("Type the model you want to use: ")
    dataset_input = input("Type the dataset you want to use: ")

    if model_input == "scikit":
        if dataset_input == "cup":
            print(f"--- Analyzing {dataset_input} ---")

            x_tr, y_tr = cup_data["train"]["numpy"]
            x_ts = cup_data["test"]["numpy"]

            svm_execution(x_tr, y_tr, x_ts, None, "cup", "regression")
            plot_results_svm("regression")
        elif dataset_input == "monk":
            for dataset_name, data in monk_datasets.items():
                print(f"--- Analyzing {dataset_name} --- ")

                x_tr, y_tr = data["train"]["numpy"]
                x_ts, y_ts = data["test"]["numpy"]

                svm_execution(x_tr,y_tr,x_ts,y_ts,dataset_name,"classification")
            plot_results_svm("classification")
    elif model_input == "pytorch":
        if dataset_input == "cup":
            print(f"--- Analyzing {dataset_input} ---")

            x_tr, y_tr = cup_data["train"]["torch"]
            x_ts = cup_data["test"]["torch"]

            nn_execution(x_tr,y_tr,x_ts,None,dataset_input,"regression")
            plot_results_nn("regression")
        elif dataset_input == "monk":
            for dataset_name, data in monk_datasets.items():
                print(f"--- Analyzing {dataset_name} --- ")

                x_tr, y_tr = data["train"]["torch"]
                x_ts, y_ts = data["test"]["torch"]

                nn_execution(x_tr, y_tr, x_ts, y_ts,dataset_name,"classification")
            plot_results_nn("classification")

    print("\nAnalisi completata!")

if __name__ == "__main__":
    main()


