import seaborn as sns
import matplotlib.pyplot as plt


def summary_dataset(file_input, file_output):

    count_class = {'gender_male': 0, 'gender_female': 0, 'bag': 0, 'hat': 0}
    count_fully_annotated = 0
    with open(file_input, 'r') as f:
        rows = f.readlines()

    results = []
    for row in rows:
        col = row.strip().split(',')  # Divide le colonne usando la virgola come separatore
        classes = col[-3:]      # Prende le ultime 3 colonne
        results.append(classes)
        if classes[0] != '-1':
            if classes[0] == '0':
                count_class['gender_male'] += 1
            else:
                count_class['gender_female'] += 1
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


    # # Scrive i risultati in un nuovo file
    # with open(file_output, 'w') as f:
    #     for r in results:
    #         f.write(','.join(r) + '\n')
    #
    # num_sample = len(results)
    # print(f'Numero totale di campioni: {num_sample}')


# Esempio di utilizzo
file_input = '../../Dataset/training_set.txt'  # Nome del file originale
file_output = '../../Dataset/class_training_set.txt'  # Nome del file di output

summary_dataset(file_input, file_output)
print(f"Le ultime 3 colonne sono state estratte e salvate in {file_output}.")
