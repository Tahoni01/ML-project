"""
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, output_file="results_plot.png"):
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())  # Ottieni le metriche dal primo modello
    num_metrics = len(metrics)

    # Organizza i dati per il plot
    data = {metric: [results[model][metric] for model in models] for metric in metrics}

    # Crea un grafico a barre per ogni metrica
    x = np.arange(len(models))  # Indici per i modelli
    width = 0.2  # Larghezza delle barre

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        offset = (i - num_metrics / 2) * width
        ax.bar(x + offset, data[metric], width, label=metric)

    # Aggiungi etichette e legenda
    ax.set_xlabel("Modelli")
    ax.set_ylabel("Valore della Metrica")
    ax.set_title("Risultati dei Modelli AI sui Dataset MONK")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Metriche")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Salva il grafico
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Grafico salvato in: {output_file}")

"""

import matplotlib.pyplot as plt
import numpy as np


def compute_overall_history(fold_histories):
    """
    Calcola la media delle metriche su tutti i fold.
    """
    overall_history = {
        "train_loss": np.mean([np.array(history["train_loss"]) for history in fold_histories], axis=0).tolist(),
        "val_loss": np.mean([np.array(history["val_loss"]) for history in fold_histories], axis=0).tolist(),
        "train_accuracy": np.mean([np.array(history["train_accuracy"]) for history in fold_histories], axis=0).tolist(),
        "val_accuracy": np.mean([np.array(history["val_accuracy"]) for history in fold_histories], axis=0).tolist(),
    }
    return overall_history


def plot_overall_history(overall_history, title="Overall Metrics", output_file=None):
    """
    Plotta le metriche medie su tutti i fold.
    """
    epochs = range(1, len(overall_history["train_loss"]) + 1)

    # Plot della Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, overall_history["train_loss"], label="Loss TR (Mean)")
    plt.plot(epochs, overall_history["val_loss"], label="Loss TS (Mean)")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot dell'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, overall_history["train_accuracy"], label="Accuracy TR (Mean)")
    plt.plot(epochs, overall_history["val_accuracy"], label="Accuracy TS (Mean)")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Grafico salvato in: {output_file}")
    else:
        plt.show()
