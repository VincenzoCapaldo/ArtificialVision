import seaborn as sns
import matplotlib.pyplot as plt
import os
import random

def split_train_validation(input_file, output_train_file, output_val_file, training_dim=0.8):
    # Leggi i dati dal file di input
    with open(input_file, 'r') as f:
        rows = f.readlines()

    # Mescola i dati casualmente
    random.shuffle(rows)

    # Determina il numero di righe per il training set
    num_training_samples = int(len(rows) * training_dim)

    # Dividi i dati
    training_data = rows[:num_training_samples]
    validation_data = rows[num_training_samples:]

    # Salva i dati nei file di output
    with open(output_train_file, 'w') as f:
        f.writelines(training_data)

    with open(output_val_file, 'w') as f:
        f.writelines(validation_data)

    print(f"Training set salvato in {output_train_file} con {len(training_data)} campioni.")
    print(f"Validation set salvato in {output_val_file} con {len(validation_data)} campioni.")




def summary_dataset(file_input, file_output):
    count_class = {'gender': 0, 'bag': 0, 'hat': 0}
    count_fully_annotated = 0
    with open(file_input, 'r') as f:
        rows = f.readlines()

    results = []
    for row in rows:
        col = row.strip().split(',')  # Divide le colonne usando la virgola come separatore
        name = col[0]
        classes = col[-3:]  # Prende le ultime 3 colonne
        res = [name] + classes
        results.append(res)
        if classes[0] != '-1':
            count_class['gender'] += 1
        if classes[1] != '-1':
            count_class['bag'] += 1
        if classes[2] != '-1':
            count_class['hat'] += 1

        if classes[0] != '-1' and classes[1] != '-1' and classes[2] != '-1':
            count_fully_annotated += 1

    num_sample = len(results)
    # Creazione del grafico a barre
    sns.set_palette(sns.dark_palette("seagreen"))
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(count_class.keys()), y=list(count_class.values()))
    # Personalizzazione del grafico
    plt.title(f'total number of samples: {num_sample}', fontsize=16)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Number of sample', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Mostra il grafico
    plt.show()

    print(f'Number of fully annotated sample: {count_fully_annotated}')

    # Scrive i risultati in un nuovo file
    with open(file_output, 'w') as f:
        for r in results:
            f.write(','.join(r) + '\n')

#
# # Esempio di utilizzo
# file_input = '../../Dataset/training_set.txt'  # Nome del file originale
# file_output = '../../training_set.txt'  # Nome del file di output
#
# summary_dataset(file_input, file_output)
#
# file_input = '../../Dataset/validation_set.txt'  # Nome del file originale
# file_output = '../../PreprocessedDataset/test_set.txt'  # Nome del file di output
# summary_dataset(file_input, file_output)
#
#
# # Esempio di utilizzo
# input_file = '../../training_set.txt'
# output_train_file = '../../PreprocessedDataset/train_split.txt'
# output_val_file = '../../PreprocessedDataset/val_split.txt'
# TRAINING_DIM = 0.8  # Percentuale di dati per il training
#
# split_train_validation(input_file, output_train_file, output_val_file, TRAINING_DIM)

with open('../../PreprocessedDataset/val_split.txt', 'r') as f:
    rows = f.readlines()

results = []
for row in rows:
    col = row.strip().split(',')  # Divide le colonne usando la virgola come separatore
    name = col[0]

    results.append(name)

with open('../../PreprocessedDataset/val_split2.txt', 'w') as f:
    for r in results:
        f.write(r + '\n')