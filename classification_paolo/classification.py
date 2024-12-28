import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parametri
img_width, img_height = 224, 224
batch_size = 32
epochs = 20  # Aumentato per permettere una migliore convergenza
learning_rate = 0.0005
num_workers = 0

# Trasformazioni per il pre-processing delle immagini
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),  # Data augmentation
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
                gender = int(parts[4])

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

# Percorsi dei dati
data_dirs = {
    "train": "dataset/training_set",
    "val": "dataset/training_set"
}
label_files = {
    "train": "dataset/train_split.txt",
    "val": "dataset/val_split.txt"
}

# Dataset e DataLoader
train_dataset = GenderDataset(data_dirs['train'], label_files['train'], transform)
val_dataset = GenderDataset(data_dirs['val'], label_files['val'], transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

# Modello
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Fine-tuning: sblocca gli ultimi layer convoluzionali
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():  # Sblocca solo l'ultimo blocco
    param.requires_grad = True

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

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

# Criterio di perdita e ottimizzatore
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Funzione di training
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')
    ep_loss = []
    ep_accuracy = []
    v_loss = []
    v_accuracy = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds

        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
        ep_loss.append(epoch_loss)
        ep_accuracy.append(epoch_accuracy)

        # Validazione
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())

                val_loss += loss.item()
                preds = (outputs > 0).float()
                val_correct_preds += (preds == labels).sum().item()
                val_total_preds += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct_preds / val_total_preds
        v_loss.append(val_loss)
        v_accuracy.append(val_accuracy)



        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Salva il miglior modello
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../best_bag_model.pth')
            print("Miglior modello salvato!")

        # Aggiorna lo scheduler
        scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(ep_loss, label="Train Loss", marker='o')
    plt.plot(v_loss, label="Validation Loss", marker='o')

    # Aggiungi titolo e etichette
    plt.title("Train and Validation Loss Over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # Aggiungi legenda
    plt.legend(fontsize=12)
    plt.grid(True)

    # Salva il grafico come immagine
    plt.savefig("loss_plot_gender.png", format='png', dpi=300, bbox_inches='tight')

    # Chiude la figura per liberare memoria (opzionale)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ep_accuracy, label="Train Accuracy", marker='o')
    plt.plot(v_accuracy, label="Validation Accuracy", marker='o')

    # Aggiungi titolo e etichette
    plt.title("Train and Validation accuracy Over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)

    # Aggiungi legenda
    plt.legend(fontsize=12)
    plt.grid(True)

    # Salva il grafico come immagine
    plt.savefig("accuracy_plot_gender.png", format='png', dpi=300, bbox_inches='tight')

    # Chiude la figura per liberare memoria (opzionale)
    plt.close()



# Esegui il training
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=epochs)
