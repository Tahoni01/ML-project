from preprocess import load_all_monks
from models.nn_pytorch import SimpleNN, TwoHiddenLayerNN, DropoutNN, k_fold_cross_validation as pytorch_kfold
from models.nn_jax import k_fold_cross_validation as jax_kfold
from models.svm_scikit_learn import k_fold_cross_validation as scikit_kfold
from models.svm_thunder import k_fold_cross_validation_thunder
from utility.plot_manager import plot_overall_history, compute_overall_history, plot_learning_curve
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

