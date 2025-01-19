"""
from preprocess import load_all_monks,load_monk_dataset
#from models.nn_pytorch import train_pytorch_nn, evaluate_pytorch_nn


def main():
    monk_train = "data/monk/monks-1.train"
    monk_test = "data/monk/monks-1.test"

    X_train, y_train, encoder = load_monk_dataset(monk_train)
    X_test, y_test, _ = load_monk_dataset(monk_test, encoder=encoder)

    input_size = X_train.shape[1]  # Numero di feature preprocessate
    hidden_size = 20  # Numero di neuroni nello strato nascosto
    epochs = 100  # Numero di epoche
    batch_size = 32  # Dimensione del batch
    lr = 0.001  # Learning rate

    model = train_pytorch_nn(X_train, y_train, input_size, hidden_size, epochs, batch_size, lr)

    evaluate_pytorch_nn(model, X_test, y_test)


if __name__ == "__main__":
    main()

"""

from preprocess import load_all_monks
from models.nn_pytorch import SimpleNN, TwoHiddenLayerNN, DropoutNN, k_fold_cross_validation as pytorch_kfold
from models.nn_jax import k_fold_cross_validation as jax_kfold
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


if __name__ == "__main__":
    main()

