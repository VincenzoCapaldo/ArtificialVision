import os
from collections import Counter
import matplotlib.pyplot as plt
import random


def calculate_class_weights_from_file(file_path='./dataset/training_set.txt'):
    """
    Calcola i pesi per ciascuna classe in base alla frequenza delle etichette, leggendo dal file di etichette.

    Args:
        file_path (str): Percorso al file che contiene le etichette.

    Returns:
        dict: Dizionario con i pesi per ciascuna classe.
    """
    # Verifica che il file esista
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste.")

    # Legge le etichette dal file
    labels = {'gender': [], 'bag': [], 'hat': []}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                # Estrae le etichette (gender, bag, hat)
                labels['gender'].append(int(parts[3]))
                labels['bag'].append(int(parts[4]))
                labels['hat'].append(int(parts[5]))

    # Calcola i pesi per ciascun task separatamente
    class_weights = {}
    for task, task_labels in labels.items():
        label_counts = Counter(task_labels)
        max_count = max(label_counts.values())
        class_weights[task] = {label: max_count / count for label, count in label_counts.items() if label != -1}

    # Logging delle statistiche
    for task, weights in class_weights.items():
        print(f"Distribuzione delle etichette per {task}: {Counter(labels[task])}")
        print(f"Pesi delle classi per {task}: {weights}")

    return class_weights, labels


def plot_label_distribution(labels, output_path='./classification_andrea/statistics/'):
    """
    Crea istogrammi per la distribuzione delle etichette (gender, bag, hat).

    Args:
        labels (dict): Dizionario con liste di etichette per ciascun task.
        output_path (str): Percorso per salvare i grafici.
    """
    os.makedirs(output_path, exist_ok=True)

    for task, task_labels in labels.items():
        label_counts = Counter(task_labels)
        labels_, counts = zip(*label_counts.items())

        plt.figure(figsize=(8, 5))
        plt.bar(labels_, counts, color='skyblue')
        plt.xlabel('Etichette')
        plt.ylabel('Conteggio')
        plt.title(f'Distribuzione delle etichette per {task}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_path, f'{task}_distribution.png'))
        plt.close()
        print(f"Grafico della distribuzione per {task} salvato in {output_path}")



def dividi_train_val(input_file, train_file, val_file, train_ratio=0.8, random_seed=None):
    """
    Divide un file di testo in due file: training e validation.

    Args:
        input_file (str): Percorso del file di input.
        train_file (str): Percorso del file di output per il training set.
        val_file (str): Percorso del file di output per il validation set.
        train_ratio (float): Proporzione di dati da assegnare al training set (default 0.8).
        random_seed (int, optional): Seed per il generatore di numeri casuali per garantire riproducibilità.
    """
    # Imposta il seed per la riproducibilità, se fornito
    if random_seed is not None:
        random.seed(random_seed)

    # Leggi tutte le righe dal file di input
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Mescola le righe in modo casuale
    random.shuffle(lines)

    # Calcola l'indice di divisione
    split_index = int(len(lines) * train_ratio)

    # Suddividi le righe in training e validation set
    train_lines = lines[:split_index]
    val_lines = lines[split_index:]

    # Scrivi le righe nei rispettivi file di output
    with open(train_file, 'w') as train_outfile:
        train_outfile.writelines(train_lines)

    with open(val_file, 'w') as val_outfile:
        val_outfile.writelines(val_lines)


if __name__ == "__main__":

    # input_file = "./dataset/training_set.txt"
    # train_file = "./dataset/train_split.txt"
    # val_file = "./dataset/val_split.txt"
    #
    # dividi_train_val(input_file, train_file, val_file, 0.8, 65464)
    # print("Training set diviso in train e validation")

    #--- TRAIN PLOT ---
    weights, labels = calculate_class_weights_from_file("./dataset/train_split.txt")

    # --- VALIDATION PLOT ---
    #weights, labels = calculate_class_weights_from_file("./dataset/val_split.txt")

    # --- TEST PLOT ---
    #weights, labels = calculate_class_weights_from_file("./dataset/test_set.txt")

    print("Pesi calcolati:", weights)
    # Crea e salva i grafici della distribuzione
    plot_label_distribution(labels)
