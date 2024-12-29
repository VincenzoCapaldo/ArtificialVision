import os
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from dataset import TrainDataset, ValidationDataset
from nets import PARMultiTaskNet
import torch
import numpy as np


def calculate_class_weights(dataset):
    """
    Calcola i pesi per bilanciare le classi per ogni task e assegna un peso per ogni campione.
    :param dataset: Dataset PyTorch
    :return: Array di pesi per ogni campione
    """
    # Calcolati da preprocess con seed=65464
    gender_dist = Counter({0: 49383, 1: 18952, -1: 6129})
    bag_dist = Counter({0: 44237, -1: 21829, 1: 8398})
    hat_dist = Counter({0: 54941, -1: 11838, 1: 7685})

    scale_factor = 1000
    gender_weights = {label: (1.0 / count) * scale_factor for label, count in gender_dist.items() if label != -1}
    bag_weights = {label: (1.0 / count) * scale_factor for label, count in bag_dist.items() if label != -1}
    hat_weights = {label: (1.0 / count) * scale_factor for label, count in hat_dist.items() if label != -1}

    sample_weights = []
    for i in range(len(dataset)):
        # Estrai le etichette del campione
        labels = np.array(dataset[i][1])

        # Calcola i pesi per ogni task, assegnando 0.0 se l'etichetta è -1
        gender_weight = gender_weights.get(labels[0], 0.0)
        bag_weight = bag_weights.get(labels[1], 0.0)
        hat_weight = hat_weights.get(labels[2], 0.0)

        # Se tutte le label sono -1, assegna peso 0.0
        if all(label == -1 for label in labels):
            combined_weight = 0.0
        else:
            # Calcola il peso combinato come media dei pesi validi
            combined_weight = np.mean([gender_weight, bag_weight, hat_weight])

        #print(labels,combined_weight)

        sample_weights.append(combined_weight)

    return np.array(sample_weights)



def initialize_weights(module):
    """
    Inizializza i pesi del modello usando Xavier Uniform.
    :param module: Modulo PyTorch da inizializzare
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def masked_loss(criterion, outputs, labels, mask):
    """
    Applica una masked loss, escludendo contributi delle label -1.
    :param criterion: Funzione di perdita (es. BCEWithLogitsLoss)
    :param outputs: Output del modello
    :param labels: Etichette target
    :param mask: Maschera binaria (1 dove label >= 0, 0 altrove)
    :return: Loss mascherata
    """
    masked_outputs = outputs[mask]
    masked_labels = labels[mask]
    if masked_outputs.numel() == 0:  # Evita errore se maschera è vuota
        return torch.tensor(0.0, device=outputs.device).mean()
    return criterion(masked_outputs, masked_labels).mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        images, labels = images.to(device), labels.to(device)
        masks = labels >= 0

        optimizer.zero_grad()
        outputs = model(images)

        gender_loss = masked_loss(criterion, outputs["gender"].squeeze(), labels[:, 0], masks[:, 0])
        bag_loss = masked_loss(criterion, outputs["bag"].squeeze(), labels[:, 1], masks[:, 1])
        hat_loss = masked_loss(criterion, outputs["hat"].squeeze(), labels[:, 2], masks[:, 2])
        loss = gender_loss + bag_loss + hat_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0

    # Metriche per ogni task
    all_labels = {"gender": [], "bag": [], "hat": []}
    all_predictions = {"gender": [], "bag": [], "hat": []}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)

            # Crea maschere per ogni task (1 dove label >= 0, 0 altrove)
            masks = labels >= 0

            outputs = model(images)
            gender_loss = masked_loss(criterion, outputs["gender"].squeeze(), labels[:, 0], masks[:, 0])
            bag_loss = masked_loss(criterion, outputs["bag"].squeeze(), labels[:, 1], masks[:, 1])
            hat_loss = masked_loss(criterion, outputs["hat"].squeeze(), labels[:, 2], masks[:, 2])

            loss = gender_loss + bag_loss + hat_loss
            running_loss += loss.item()

            # Predizioni e etichette
            for task in ["gender", "bag", "hat"]:
                preds = torch.sigmoid(outputs[task]) > 0.5
                all_predictions[task].extend(preds[masks[:, ["gender", "bag", "hat"].index(task)]].cpu().numpy())
                task_index = ["gender", "bag", "hat"].index(task)
                all_labels[task].extend(labels[masks[:, task_index], task_index].cpu().numpy())

    # Calcolo delle metriche per ogni task
    metrics = {}
    for task in ["gender", "bag", "hat"]:
        accuracy = accuracy_score(all_labels[task], all_predictions[task])
        precision = precision_score(all_labels[task], all_predictions[task], zero_division=0)
        recall = recall_score(all_labels[task], all_predictions[task], zero_division=0)
        f1 = f1_score(all_labels[task], all_predictions[task], zero_division=0)
        metrics[task] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return running_loss / len(dataloader), metrics


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
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path al dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization)')
    parser.add_argument('--epochs', type=int, default=50, help='Numero di epoche')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda o cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='./classification_andrea/checkpoints',
                        help='Directory dei checkpoint')
    parser.add_argument('--resume_checkpoint', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--backbone', type=str, default='resnet18modifiche tra', help='Backbone name: resnet50 o resnet18')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Usando il device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Istanziando training set...")
    train_dataset = TrainDataset(data_dir=args.data_dir)
    print("Istanziando validation set...")
    val_dataset = ValidationDataset(data_dir=args.data_dir)

    print("Calcolo dei pesi delle classi...")
    class_weights = calculate_class_weights(train_dataset)
    sampler = WeightedRandomSampler(class_weights, len(train_dataset))

    print("Istanziando dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = PARMultiTaskNet(backbone_name=args.backbone, pretrained=True).to(device)
    # Applica l'inizializzazione ai soli moduli delle teste
    model.gender_head.apply(initialize_weights)
    model.bag_head.apply(initialize_weights)
    model.hat_head.apply(initialize_weights)

    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Usa riduzione 'none' per supportare la masked loss
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},  # Backbone con learning rate ridotto
        {'params': model.gender_head.parameters()},  # Testa gender
        {'params': model.bag_head.parameters()},  # Testa bag
        {'params': model.hat_head.parameters()}  # Testa hat
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Aggiungi lo scheduler
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = []
    loss_history = []
    val_loss_history = []

    print("Inizio train...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
        metrics_history.append(val_metrics)
        loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch [{epoch + 1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        for task, metrics in val_metrics.items():
            print(
                f"{task.capitalize()} - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f},Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        plot_metrics(metrics_history, args.checkpoint_dir, epoch + 1, loss_history, val_loss_history)

        # Passa la perdita di validazione allo scheduler
        #scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Salvato il modello migliore in {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    main()
