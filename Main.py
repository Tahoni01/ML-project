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
