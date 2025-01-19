"""
from preprocess import load_all_monks
from models.nn_pytorch import SimpleNN, TwoHiddenLayerNN, DropoutNN, k_fold_cross_validation
from utility.plot_manager import plot_results



def main():
    print("Caricamento dei dati...")
    base_path = "data/monk/"
    datasets = load_all_monks(base_path)

    # Parametri comuni
    input_size = next(iter(datasets.values()))["X_train"].shape[1]
    hidden_size = 20
    epochs = 50
    batch_size = 32
    lr = 0.001
    k = 5  # Numero di fold per la K-Fold Cross-Validation

    results = {}  # Dizionario per salvare i risultati medi dei modelli

    for monk_name, data in datasets.items():
        print(f"\n--- Validazione K-Fold per {monk_name} ---")

        # SimpleNN
        print("\nModello: SimpleNN")
        metrics_simple = k_fold_cross_validation(
            data["X_train"], data["y_train"], SimpleNN, input_size, hidden_size, k, epochs, batch_size, lr
        )
        results[f"{monk_name}_SimpleNN"] = metrics_simple

        # TwoHiddenLayerNN
        print("\nModello: TwoHiddenLayerNN")
        metrics_two_hidden = k_fold_cross_validation(
            data["X_train"], data["y_train"], TwoHiddenLayerNN, input_size, hidden_size, k, epochs, batch_size, lr
        )
        results[f"{monk_name}_TwoHiddenLayerNN"] = metrics_two_hidden

        # DropoutNN
        print("\nModello: DropoutNN")
        metrics_dropout = k_fold_cross_validation(
            data["X_train"], data["y_train"], DropoutNN, input_size, hidden_size, k, epochs, batch_size, lr
        )
        results[f"{monk_name}_DropoutNN"] = metrics_dropout

    # Dopo il training e la validazione, salva il grafico
    plot_results(results, output_file="results_plot.png")


if __name__ == "__main__":
    main()
"""

from preprocess import load_all_monks
from models.nn_pytorch import SimpleNN, TwoHiddenLayerNN, DropoutNN, k_fold_cross_validation as pytorch_kfold
from models.nn_jax import k_fold_cross_validation as jax_kfold
from models.svm_scikit_learn import k_fold_cross_validation as svm_kfold
from utility.plot_manager import plot_overall_history, compute_overall_history
import os


def main():
    print("Caricamento dei dati...")
    base_path = "data/monk/"
    datasets = load_all_monks(base_path)

    # Parametri comuni
    input_size = next(iter(datasets.values()))["X_train"].shape[1]
    hidden_size = 20
    epochs = 50
    batch_size = 32
    lr = 0.001
    k = 5  # Numero di fold per la K-Fold Cross-Validation

    # Directory per salvare i risultati
    result_dir = "results/"
    os.makedirs(result_dir, exist_ok=True)

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

        # scikit-learn
        print("\nModello SVM")
        svm_histories, svm_overall = svm_kfold(data["X_train"], data["y_train"], kernel='rbf', C=1.0, k=k)
        print(f"Risultati SVM per {monk_name}: {svm_overall}")
        # Plotta il grafico per SVM
        """
        plot_overall_history(
            compute_overall_history(svm_histories),
            title=f"SVM - {monk_name} - Overall Metrics",                     #DA sistemare
            output_file=f"{result_dir}{monk_name}_SVM_Overall.png"
        )"""


if __name__ == "__main__":
    main()
