from preprocess import load_all_monks
from models.nn_pytorch import k_fold_cross_validation

def main():
    # Percorso ai dataset MONK
    base_path = "data/monk/"

    # Carica i dataset
    datasets = load_all_monks(base_path)

    # Parametri del modello
    hidden_size = 10
    epochs = 100
    batch_size = 32
    lr = 0.001
    k = 5  # Numero di fold

    # Esegui la K-fold cross-validation per ogni dataset
    for monk_name, data in datasets.items():
        print(f"\n--- Validazione K-fold per {monk_name} ---")
        X, y = data["X_train"], data["y_train"]
        input_size = X.shape[1]

        accuracies = k_fold_cross_validation(X, y, input_size, hidden_size, k, epochs, batch_size, lr)
        mean_accuracy = sum(accuracies) / len(accuracies)

        print(f"Accuratezze per {monk_name}: {accuracies}")
        print(f"Accuratezza media per {monk_name}: {mean_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
