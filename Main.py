import time
from preprocess import load_all_monks, load_cup_data
from models.svm_scikit_learn import svm_execution, plot_results_svm
from models.nn_pytorch import nn_execution, plot_results_nn
import os

def main():
    print("Loading data...")

    # Define dataset paths
    base_path_monk = "data/monk/"
    cup_train_path = "data/cup/ML-CUP24-TR.csv"
    cup_test_path = "data/cup/ML-CUP24-TS.csv"

    # Load datasets
    monk_datasets, encoders, scalers = load_all_monks(base_path_monk)
    cup_data = load_cup_data(cup_train_path, cup_test_path)

    # Create  results directory if it does not exist
    result_dir = "results/"
    os.makedirs(result_dir, exist_ok=True)

    # User input for model and dataset selection
    model_input = input("Enter the model to use (scikit/pytorch): ")
    dataset_input = input("Enter the dataset to use (monk/cup): ")

    # Execute SVM model using Scikit-Learn
    if model_input == "scikit":
        if dataset_input == "cup":
            print(f"--- Analyzing {dataset_input} ---")

            x_tr, y_tr = cup_data["train"]["numpy"]
            x_ts = cup_data["test"]["numpy"]

            start = time.time()
            svm_execution(x_tr, y_tr, x_ts, None, "cup", "regression")
            elapsed_time = time.time() - start
            print(f"⏳ Time taken for SVM execution (CUP): {elapsed_time:.2f} seconds")
            plot_results_svm("regression")

        elif dataset_input == "monk":
            for dataset_name, data in monk_datasets.items():
                print(f"--- Analyzing {dataset_name} --- ")

                x_tr, y_tr = data["train"]["numpy"]
                x_ts, y_ts = data["test"]["numpy"]
                start = time.time()
                svm_execution(x_tr,y_tr,x_ts,y_ts,dataset_name,"classification")
                elapsed_time = time.time() - start
                print(f"⏳ Time taken for SVM execution (monk): {elapsed_time:.2f} seconds")
            plot_results_svm("classification")

    # Execute Neural Network model using PyTorch
    elif model_input == "pytorch":
        if dataset_input == "cup":
            print(f"--- Analyzing {dataset_input} ---")

            x_tr, y_tr = cup_data["train"]["torch"]
            x_ts = cup_data["test"]["torch"]

            start = time.time()
            nn_execution(x_tr,y_tr,x_ts,None,dataset_input,"regression")
            elapsed_time = time.time() - start
            print(f"⏳ Time taken for NN execution (CUP): {elapsed_time:.2f} seconds")
            plot_results_nn("regression")

        elif dataset_input == "monk":
            for dataset_name, data in monk_datasets.items():
                print(f"--- Analyzing {dataset_name} --- ")

                x_tr, y_tr = data["train"]["torch"]
                x_ts, y_ts = data["test"]["torch"]

                start = time.time()
                nn_execution(x_tr, y_tr, x_ts, y_ts,dataset_name,"classification")
                elapsed_time = time.time() - start
                print(f"⏳ Time taken for NN execution (CUP): {elapsed_time:.2f} seconds")

            plot_results_nn("classification")

    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()


