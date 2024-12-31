import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class PARCustomDataset(Dataset):
    def __init__(self, data_dir="./dataset", txt_file="training_set.txt", transforms=None):
        """
        Classe base per gestire il dataset di PAR.
        :param data_dir: Path alla directory delle immagini (default: "./dataset")
        :param txt_file: File di testo con le annotazioni (default: "training_set.txt")
        :param transforms: Trasformazioni da applicare alle immagini
        """
        self.data_dir = data_dir
        self.txt_file = txt_file
        self.transforms = transforms

        # Carica i dati dal file txt
        self.data = []
        with open(os.path.join(data_dir, txt_file), 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split(',')
                    img_name, gender, bag, hat = parts[0], int(parts[3]), int(parts[4]), int(parts[5])
                    img_subdir = "test_set" if "test_set" in txt_file else "training_set"
                    img_path = os.path.join(data_dir, img_subdir, img_name)

                    # Assicurati che l'immagine esista
                    if os.path.exists(img_path):
                        self.data.append((img_path, [gender, bag, hat]))
                    else:
                        print(f"Immagine mancante: {img_path}")
                except Exception as e:
                    print(f"Errore nel parsing della linea: {line}. Dettagli: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Restituisce un elemento dal dataset.
        :param index: Indice dell'elemento
        :return: Tuple (immagine, label)
        """
        try:
            img_path, labels = self.data[index]
            image = Image.open(img_path).convert('RGB')
            if self.transforms:
                image = self.transforms(image)
            return image, torch.tensor(labels, dtype=torch.float32)
        except Exception as e:
            print(f"Errore durante il caricamento dell'immagine o delle etichette per l'indice {index}. Dettagli: {e}")
            return None, None


class TrainDataset(PARCustomDataset):
    def __init__(self, data_dir="./dataset", txt_file="train_split.txt"):
        """
        Classe per il dataset di training con split train/val.
        :param data_dir: Path alla directory delle immagini (default: "./dataset")
        :param txt_file: File di testo con le annotazioni (default: "training_set.txt")
        :param val_ratio: Percentuale di dati per la validazione
        :param train: True per il set di training, False per il set di validazione
        """
        # Trasformazioni specifiche per train e validation
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        super().__init__(data_dir, txt_file, transforms)


class ValidationDataset(PARCustomDataset):
    def __init__(self, data_dir="./dataset", txt_file="val_split.txt"):
        """
        Classe per il dataset di training con split train/val.
        :param data_dir: Path alla directory delle immagini (default: "./dataset")
        :param txt_file: File di testo con le annotazioni (default: "training_set.txt")
        :param val_ratio: Percentuale di dati per la validazione
        :param train: True per il set di training, False per il set di validazione
        """
        # Trasformazioni specifiche per train e validation
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        super().__init__(data_dir, txt_file, transforms)


class TestDataset(PARCustomDataset):
    def __init__(self, data_dir="./dataset", txt_file="test_set.txt"):
        """
        Classe per il dataset di test.
        :param data_dir: Path alla directory delle immagini (default: "./dataset")
        :param txt_file: File di testo con le annotazioni (default: "test_set.txt")
        """
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        super().__init__(data_dir, txt_file, transforms)


if __name__ == "__main__":
    # Test delle classi
    data_dir = "./dataset"

    # Dataset di training
    train_dataset = TrainDataset(data_dir=data_dir)
    val_dataset = ValidationDataset(data_dir=data_dir)
    test_dataset = TestDataset(data_dir=data_dir)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Esempio di accesso ai dati
    for i in range(len(train_dataset)):
        image, labels = train_dataset[i]
        if image is not None and labels is not None:
            print(f"Immagine shape: {image.shape}, Labels: {labels}")
