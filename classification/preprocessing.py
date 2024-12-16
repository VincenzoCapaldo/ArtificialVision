import os
from PIL import Image

def resize_images(base_path="dataset/", size=(256, 256)):
    """
    Ridimensiona tutte le immagini nelle cartelle 'training_set' e 'test_set' a 256x256 pixel,
    sovrascrivendole con gli stessi nomi nei rispettivi percorsi.

    Parametri:
    ----------
    base_path : str
        Percorso base della directory contenente 'training_set' e 'test_set'.
    size : tuple, opzionale
        Nuova dimensione delle immagini (default: (256, 256)).
    """
    # Percorsi delle sottocartelle
    training_path = os.path.join(base_path, "training_set")
    test_path = os.path.join(base_path, "test_set")

    # Funzione per ridimensionare le immagini in una directory
    def resize_in_directory(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                # Controlla che il file sia un'immagine (estensioni comuni)
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(root, file)  # Percorso completo del file
                    try:
                        # Apre l'immagine
                        img = Image.open(full_path)
                        # Ridimensiona l'immagine
                        img_resized = img.resize(size, Image.ANTIALIAS)
                        # Salva l'immagine ridimensionata nello stesso percorso
                        img_resized.save(full_path)
                    except Exception as e:
                        print(f"Errore con il file {full_path}: {e}")

    # Ridimensiona le immagini nelle cartelle di training e test
    print("Inizio ridimensionamento delle immagini nel training set...")
    resize_in_directory(training_path)

    print("Inizio ridimensionamento delle immagini nel test set...")
    resize_in_directory(test_path)

    print("Ridimensionamento completato.")

# DA FARE FUNZIONE CHE DIVIDE IN TRAIN E VALIDATION

# def save2lists(im_list, att_list, filename):
#     L = []
#     for c, im in enumerate(im_list):
#         tmp = list(att_list[c])
#         L.append([str(c)]+map(str,tmp)+[str(im)])
#     with open(filename, 'w') as f:
#         writer = csv.writer(f, delimiter='\t')
#         writer.writerows(L)
#
# def data_prep(full_path):
#     im_list_tr, att_list_tr, im_list_val, att_list_val, im_list_test, att_list_test = get_data(full_path)
#     save2lists(im_list_tr, att_list_tr,'training_list.lst')
#     save2lists(im_list_val, att_list_val,'valid_list.lst')
#     save2lists(im_list_test, att_list_test,'testing_list.lst')
