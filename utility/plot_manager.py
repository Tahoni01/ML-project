import matplotlib.pyplot as plt

def plot_training_curves(history, result_dir, file_prefix):
    """
    Plotta le curve di training e validazione per loss e accuracy.

    Args:
        history (dict): Dizionario contenente "train_loss", "val_loss", "train_accuracy", "val_accuracy".
        result_dir (str): Directory in cui salvare i grafici.
        file_prefix (str): Prefisso del nome dei file.
    """
    # Plot delle curve di perdita
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{result_dir}/{file_prefix}_loss_curves.png")
    plt.close()

    # Plot delle curve di accuratezza
    plt.figure()
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{result_dir}/{file_prefix}_accuracy_curves.png")
    plt.close()
