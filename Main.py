"""
from preprocess import load_all_monks, load_monk_data
from models.nn_pytorch import optimize_hyperparameters, final_evaluation
from models.nn_jax import k_fold_cross_validation as jax_kfold
from models.svm_scikit_learn import k_fold_cross_validation as scikit_kfold
from models.svm_thunder import k_fold_cross_validation_thunder
#from utility.plot_manager import plot_overall_history, compute_overall_history, plot_learning_curve
import os
import numpy as np


def main():
    print("Caricamento dei dati...")
    base_path = "data/monk/"
    datasets = load_all_monks(base_path)
    x,y = load_monk_data("data/monk/monks-3.train")
    z,w = load_monk_data('data/monk/monks-3.test')

    input_size = x.shape[1]
    output_size = len(np.unique(y))

    # Directory per salvare i risultati
    result_dir = "results/"
    os.makedirs(result_dir, exist_ok=True)

    param_bounds = {
        "hidden_size": (16, 100),
        "lr": (1e-4, 0.01),
        "weight_decay": (1e-5, 1e-3),  # Range tipico per weight decay
    }
    epochs = 500
    k = 5
    device = "cpu"

    model, best_params, best_val_accuracy = optimize_hyperparameters(x, y, input_size, param_bounds, epochs, k, device)

    final_evaluation(model, x, y, z, w, device)

    print(f"Best Parameters: {best_params} with Val Accuracy: {best_val_accuracy:.4f}")





    # Parametri comuni
    input_size = next(iter(datasets.values()))["X_train"].shape[1]
    hidden_size = 20
    epochs = 50
    batch_size = 32
    lr = 0.001
    k = 5  # Numero di fold per la K-Fold Cross-Validation

    # Modelli PyTorch e JAX
    pytorch_models = {
        "SimpleNN": SimpleNN,
        "TwoHiddenLayerNN": TwoHiddenLayerNN,
        "DropoutNN": DropoutNN
    }

    for monk_name, data in datasets.items():
        print(f"\n--- Validazione K-Fold per {monk_name} ---")

        for model_name, model_class in pytorch_models.items():
            # PyTorch
            print(f"\nModello PyTorch: {model_name}")
            pytorch_histories = pytorch_kfold(
                data["X_train"], data["y_train"], model_class, input_size, hidden_size, k, epochs, batch_size, lr
            )
            pytorch_overall = compute_overall_history(pytorch_histories)
            plot_overall_history(
                pytorch_overall,
                title=f"PyTorch - {monk_name} - {model_name} - Overall Metrics",
                output_file=f"{result_dir}{monk_name}_PyTorch_{model_name}_Overall.png"
            )

        # JAX
        print("\nModello JAX")
        jax_histories, jax_overall = jax_kfold(
            data["X_train"], data["y_train"], input_size, hidden_size, k, epochs, lr
        )
        plot_overall_history(
            compute_overall_history(jax_histories),
            title=f"JAX - {monk_name} - Overall Metrics",
            output_file=f"{result_dir}{monk_name}_JAX_Overall.png"
        )

        # Esegui il test per scikit
        print("\nModello Scikit")
        scikit_histories, scikit_overall, scikit_learning_data = scikit_kfold(
            data["X_train"], data["y_train"], kernel='poly', C=1.0, k=k
        )

        # Plotta la curva di apprendimento
        plot_learning_curve(
            scikit_learning_data["train_sizes"],
            scikit_learning_data["train_loss"],
            scikit_learning_data["val_loss"],
            title=f"Scikit - {monk_name} - Learning Curve",
            output_file=f"{result_dir}{monk_name}_Scikit_LearningCurve.png"
        )

        # ThunderSVM
        print("\nModello Thunder")
        thunder_histories, thunder_overall, thunder_learning_data = k_fold_cross_validation_thunder(
            data["X_train"], data["y_train"], k=k, kernel='rbf', C=1.0, gamma='scale'
        )

        # Plotta la curva di apprendimento per ThunderSVM
        plot_learning_curve(
            thunder_learning_data["train_sizes"],
            thunder_learning_data["train_loss"],
            thunder_learning_data["val_loss"],
            title=f"ThunderSVM - {monk_name} - Learning Curve",
            output_file=f"{result_dir}{monk_name}_Thunder_LearningCurve.png"
        )

if __name__ == "__main__":
    main()
"""

from preprocess import load_all_monks, load_monk_data
from models.nn_pytorch import optimize_hyperparameters,final_evaluation
from models.svm_scikit_learn import svm_execution
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
        svm_execution(x,y,z,w)


    print("\nAnalisi completata per tutti i dataset Monk!")

if __name__ == "__main__":
    main()