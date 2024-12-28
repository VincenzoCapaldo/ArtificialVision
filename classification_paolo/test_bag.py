import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Trasformazioni per il pre-processing delle immagini
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset personalizzato
class GenderDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform
        self.image_labels = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                filename = parts[0]
                gender = int(parts[4])  # Il quarto valore è il genere

                if gender in [0, 1]:
                    image_path = os.path.join(self.image_dir, filename)
                    if os.path.exists(image_path):
                        self.image_labels.append((filename, gender))
                    else:
                        print(f"File non trovato, ignorato: {image_path}")

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        filename, label = self.image_labels[idx]
        image_path = os.path.join(self.image_dir, filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"File non trovato durante il caricamento: {image_path}")
            return None

        if self.transform:
            image = self.transform(image)

        return image, label

# Funzione per calcolare le metriche
def calculate_metrics(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)

    # Calcolo delle metriche
    accuracy = (torch.tensor(all_predictions) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Percorsi dei dati
    image_dir = "dataset/test_set"
    label_file = "dataset/test_set.txt"

    # Caricamento del dataset e DataLoader
    dataset = GenderDataset(image_dir, label_file, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    # Modello caricato
    model = models.resnet18(weights=None)  # Carichiamo una ResNet senza pesi pre-addestrati

    # Modifica del layer finale
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )
    model_path = 'best_bag_model.pth'  # Sostituisci con il percorso del modello salvato
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Imposta il modello in modalità valutazione

    # Calcolo delle metriche
    accuracy, precision, recall, f1 = calculate_metrics(model, dataloader)

    print(f"Accuratezza: {accuracy:.4f}")
    print(f"Precisione: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
