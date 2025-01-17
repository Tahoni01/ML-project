from preprocess import load_all_monks
from models.nn_pytorch import train_pytorch_nn, evaluate_pytorch_nn
#from models.nn_jax import JaxNN          # Modulo JAX (da implementare)

def main():
    # Percorso alla directory Monk's
    monks_path = "data/monk"

    print("Caricando tutti i dataset Monk's...")
    datasets = load_all_monks(monks_path)

    for monk, data in datasets.items():
        print(f"{monk}:")
        print(f" - Training set: {data['X_train'].shape}, Target: {data['y_train'].shape}")
        print(f" - Test set: {data['X_test'].shape}, Target: {data['y_test'].shape}")

    print("\nTutti i dataset Monk's sono stati caricati con successo!")

    # Configurazione del modello
    hidden_size = 10  # Numero di neuroni nello strato nascosto
    epochs = 50  # Numero di epoche
    batch_size = 32  # Dimensione del batch
    lr = 0.001  # Learning rate

    results = []

    # Addestra e valuta il modello per ciascun dataset
    for monk, data in datasets.items():
        print(f"\n--- Dataset {monk} ---")
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Addestramento del modello
        print(f"Addestramento del modello per {monk}...")
        model = train_pytorch_nn(
            X_train, y_train,
            input_size=X_train.shape[1],
            hidden_size=hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

        # Valutazione del modello
        print(f"Valutazione del modello per {monk}...")
        accuracy = evaluate_pytorch_nn(model, X_test, y_test)
        results.append((monk, accuracy))

    # Riassunto dei risultati
    print("\n--- Risultati Finali ---")
    for monk, accuracy in results:
        print(f"{monk}: Accuracy = {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

