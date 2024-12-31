import os
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
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


def validate(model, dataloader, device, epoch, weights):
    model.eval()
    val_losses = []
    # Metriche per ogni task
    all_labels = {"gender": [], "bag": [], "hat": []}
    all_predictions = {"gender": [], "bag": [], "hat": []}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}"):
            images, labels = images.to(device), [label.to(device) for label in labels]

            outputs = model(images)
            loss = model.compute_loss(outputs, labels)
            total_loss = sum(loss)
            loss = torch.stack(loss, dim=0)
            weighted_validation_loss = weights.to(device) @ loss.to(device)
            val_losses.append(weighted_validation_loss)

            # Predizioni e etichette
            for task in ["gender", "bag", "hat"]:
                task_index = ["gender", "bag", "hat"].index(task)
                # Crea la maschera per valori validi (label diversa da -1)
                mask = (labels[task_index] != -1)
                # Predizioni e label validi
                preds = (torch.sigmoid(outputs[task]) > 0.5).int()
                valid_preds = preds[mask]  # Filtra solo i valori validi
                valid_labels = labels[task_index][mask]  # Filtra solo i valori validi
                # Aggiungi alle liste globali
                all_predictions[task].extend(valid_preds.cpu().numpy())
                all_labels[task].extend(valid_labels.cpu().numpy())

    # Calcolo delle metriche per ogni task
    metrics = {}
    for task in ["gender", "bag", "hat"]:
        accuracy = accuracy_score(all_labels[task], all_predictions[task])
        precision = precision_score(all_labels[task], all_predictions[task], zero_division=0)
        recall = recall_score(all_labels[task], all_predictions[task], zero_division=0)
        f1 = f1_score(all_labels[task], all_predictions[task], zero_division=0)
        metrics[task] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return torch.stack(val_losses).mean().item(), metrics


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


def start_training(model, train_loader, val_loader, best_val_loss, optimizer, device, start_epoch, epochs, checkpoint_dir, patience, weights=None, l0=None):
    patience_counter = 0
    metrics_history = []
    loss_history = []
    val_loss_history = []

    # GradNorm Init
    alpha = 0.6
    weighted_loss = None
    log_weights = []
    log_loss = []
    if weights is not None:
        T = weights.sum().detach()
        # set optimizer for weights
        optimizer2 = torch.optim.Adam([weights], lr=0.01)
        init_weights = False
    else:
        optimizer2 = None
        T = None
        init_weights = True
    # --- TRAINING ---
    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            images, labels = images.to(device), [label.to(device) for label in labels]

            outputs = model(images)
            loss = model.compute_masked_loss(outputs, labels)
            #print(loss)
            total_loss = sum(loss)
            #print(total_loss)
            loss = torch.stack(loss, dim=0)

            if init_weights:
                print("Inizializzazione pesi di GradNorm...")
                weights = torch.ones_like(loss) / 3
                weights = torch.nn.Parameter(weights)
                T = weights.sum().detach()  # sum of weights
                # Optimizer2 -> for weights
                optimizer2 = torch.optim.Adam([weights], lr=0.01)
                l0 = loss.detach()  # set L(0)
                init_weights = False

            # compute the weighted loss
            weighted_loss = weights.to(device) @ loss.to(device)
            # clear gradients of network
            optimizer.zero_grad()
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(loss)):
                parameters = None
                if i == 0:
                    parameters = model.gender_head.parameters()
                elif i == 1:
                    parameters = model.bag_head.parameters()
                elif i == 2:
                    parameters = model.hat_head.parameters()
                dl = torch.autograd.grad(weights[i] * loss[i], parameters, retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)
            # compute loss ratio per task
            loss_ratio = loss.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()
            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of weights
            optimizer2.zero_grad()
            # backward pass for GradNorm
            gradnorm_loss.backward()

            # update model weights
            optimizer.step()
            # update loss weights
            optimizer2.step()
            # renormalize weights
            weights = (weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            optimizer2 = torch.optim.Adam([weights], lr=0.01)

            running_loss += total_loss.item()

            # For Plots
            # weight for each task
            log_weights.append(weights.detach().cpu().numpy().copy())
            # task normalized loss
            log_loss.append(loss_ratio.detach().cpu().numpy().copy())

        loss_history.append(running_loss / len(train_loader))

        # --- VALIDATION ---
        weighted_val_loss, val_metrics = validate(model, val_loader, device, epoch)
        metrics_history.append(val_metrics)
        val_loss_history.append(weighted_val_loss)

        # --- PLOTTING METRICS  ---
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {running_loss:.4f}, Val Loss: {weighted_val_loss:.4f}")
        for task, metrics in val_metrics.items():
            print(
                f"{task.capitalize()} - Validation Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f},Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        plot_metrics(metrics_history, checkpoint_dir, epoch + 1, loss_history, val_loss_history)

        # --- SAVING MODELS ---
        if weighted_val_loss < best_val_loss:
            best_val_loss = weighted_val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'weights': weights,
                'l0': l0
            }
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Salvato il modello migliore in {checkpoint_path}")
        else:
            patience_counter += 1

        # --- EARLY STOPPING ---
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return log_weights, log_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path al dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization)')
    parser.add_argument('--epochs', type=int, default=30, help='Numero di epoche')
    parser.add_argument('--lr_backbone', type=float, default=0.0001, help='Learning rate backbone')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate classification heads')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda o cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='./classification2_andrea/checkpoints',
                        help='Directory dei checkpoint')
    parser.add_argument('--resume_checkpoint', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone name: resnet50 o resnet18')
    parser.add_argument('--balancing', type=bool, default=False, help='Balancing batches')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--optimizer', type=str, default="adam", help='adam o sgd')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Usando il device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Istanziando training set...")
    train_dataset = TrainDataset(data_dir=args.data_dir)
    print("Istanziando validation set...")
    val_dataset = ValidationDataset(data_dir=args.data_dir)

    if args.balancing:
        print("Calcolo dei pesi delle classi per il bilanciamento dei batches...")
        class_weights = calculate_class_weights(train_dataset)
        sampler = WeightedRandomSampler(class_weights, len(train_dataset))
    else:
        sampler = SubsetRandomSampler(list(range(len(train_dataset))))

    print("Istanziando dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Istanziando modello")
    model = PARMultiTaskNet(backbone=args.backbone).to(device)

    if args.optimizer == "adam":
        optimizer = AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": model.gender_head.parameters(), "lr": args.lr},
                {"params": model.bag_head.parameters(), "lr": args.lr},
                {"params": model.hat_head.parameters(), "lr": args.lr}
            ]
        )
    else:
        optimizer = optim.SGD([
            {'params': model.backbone.parameters(), "lr": args.lr_backbone},  # Backbone con learning rate ridotto
            {'params': model.gender_head.parameters(), "lr": args.lr},  # Testa gender
            {'params': model.bag_head.parameters(), "lr": args.lr},  # Testa bag
            {'params': model.hat_head.parameters(), "lr": args.lr}  # Testa hat
        ], momentum=args.momentum, weight_decay=args.weight_decay)

    # Riprende dal checkpoint
    if args.resume_checkpoint:
        checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")]
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files, key=lambda f: os.path.getctime(os.path.join(args.checkpoint_dir, f))
            )
            checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
            print(f"Caricamento del modello dal checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            weights = checkpoint['weights']
            l0 = checkpoint['l0'].to(device)
        else:
            print("Nessun checkpoint trovato. Avvio da zero.")
            start_epoch = 0
            best_val_loss = 1e8
            weights = None
            l0 = None
    else:
        start_epoch = 0
        best_val_loss = 1e8
        weights = None
        l0 = None

    print("Inizio train...")
    log_weights, log_loss = start_training(model, train_loader, val_loader, best_val_loss, optimizer, device, start_epoch, args.epochs, args.checkpoint_dir, args.patience, weights, l0)


    # --- PLOT ---
    sigmas = [48,3,54]
    plt.figure(figsize=(16, 12))
    for i in range(len(sigmas)):
        plt.plot(log_weights[:, i], lw=3, label="σ = {}".format(sigmas[i]))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 28000)
    plt.xlabel("Iters", fontsize=18)
    plt.ylabel("Weights", fontsize=18)
    plt.title("Adaptive Weights During Training for Each Task", fontsize=18)
    plt.legend(fontsize=12)
    plt.show()

    plt.figure(figsize=(16, 12))
    plt.plot(log_loss.mean(axis=1) * 100, lw=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Iters", fontsize=18)
    plt.ylabel("Task-Normalized Error (%)", fontsize=18)
    plt.title("Training Curve of NormGrad Algorithm", fontsize=18)
    plt.show()

if __name__ == "__main__":
    main()
