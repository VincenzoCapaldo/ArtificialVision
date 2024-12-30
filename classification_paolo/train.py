import argparse
import os
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from dataset import CustomDataset
from nets import ClassificationModel

def plot_metrics(metrics_history, output_dir, epoch, loss_history, val_loss_history):
    for task in metrics_history[0].keys():
        plt.figure()
        for metric in ["accuracy", "precision", "recall", "f1"]:
            plt.plot(
                [epoch_metrics[task][metric] for epoch_metrics in metrics_history], label=f"{task}_{metric}"
            )
        plt.title(f"{task.capitalize()} Metrics (up to Epoch {epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{task}_metrics.png"))
        plt.close()

    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Loss")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "val_loss.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization)')
    parser.add_argument('--epochs', type=int, default=10, help='Numero di epoche')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda o cpu')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone name: resnet50 o resnet18')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--type_classifier', type=str, default="gender", help='gender,hat,bag')
    parser.add_argument('--balance', type=bool, default=False, help='True if balance the dataset')
    args = parser.parse_args()

    # Percorsi dei dati
    data_dirs = {
        "train": "dataset/training_set",
        "val": "dataset/training_set"
    }
    label_files = {
        "train": "dataset/train_split.txt",
        "val": "dataset/val_split.txt"
    }

    if args.type_classifier not in ["gender", "bag", "hat"]:
        print("Errore, il tipo di classificatore deve essere gender bag o hat")
        return -1

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Usando il device: {device}")

    # Modello
    model = ClassificationModel(args.backbone)
    model = model.to(device)

    # Dataset
    train_dataset = CustomDataset(data_dirs['train'], label_files['train'], type=args.type_classifier)
    val_dataset = CustomDataset(data_dirs['val'], label_files['val'], type=args.type_classifier)

    # Creazione di pesi per il sampler
    if args.balance:
        print("Bilanciando dataset...")
        labels = [label for _, label in train_dataset.image_labels]
        class_counts = Counter(labels)
        total_samples = len(labels)
        class_weights = {label: total_samples / count for label, count in class_counts.items()}

        # Assegna un peso a ogni campione
        sample_weights = [class_weights[label] for label in labels]

        # Sampler per il DataLoader
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                drop_last=True)
    else:
        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                drop_last=True)

    # Criterio di perdita e ottimizzatore
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Training
    print(f"Start training for {args.type_classifier} classifier")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, device=device,
                type=args.type_classifier, patience=args.patience)


# Funzione di training
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, type, patience):
    best_val_loss = float('inf')
    metrics_history = []
    loss_history = []
    val_loss_history = []
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # Validazione
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0
        val_metrics = {}
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())

                val_loss += loss.item()

                # Confronta le probabilità con 0.5 per ottenere le predizioni
                preds = (torch.sigmoid(outputs) > 0.5).int()

                # Calcola l'accuratezza
                val_correct_preds += (preds == labels).sum().item()
                val_total_preds += labels.size(0)

                # Salva predizioni e verità per metriche
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        val_accuracy = val_correct_preds / val_total_preds
        val_precision = precision_score(y_true_val, y_pred_val, zero_division=0)
        val_recall = recall_score(y_true_val, y_pred_val, zero_division=0)
        val_f1 = f1_score(y_true_val, y_pred_val, zero_division=0)
        val_metrics[type] = {"accuracy": val_accuracy, "precision": val_precision, "recall": val_recall, "f1": val_f1}
        metrics_history.append(val_metrics)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(
            f"Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

        plot_metrics(metrics_history, f'./classification_paolo/models', epoch + 1, loss_history, val_loss_history)

        # Salva il miglior modello
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./classification_paolo/models/best_{type}_model.pth')
            print("Miglior modello salvato!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        # Aggiorna lo scheduler
        scheduler.step()


# Esegui il training
if __name__ == '__main__':
    main()
