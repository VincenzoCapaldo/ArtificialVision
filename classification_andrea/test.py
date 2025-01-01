import os
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TestDataset
from nets import PARMultiTaskNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Funzione per testare il modello
def test_model(model, dataloader, device):
    model.eval()

    all_labels = {"gender": [], "bag": [], "hat": []}
    all_predictions = {"gender": [], "bag": [], "hat": []}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Test..."):
            images, labels = images.to(device), labels.to(device)

            # Maschera per campioni validi (etichette diverse da -1)
            masks = labels >= 0

            outputs = model(images)

            # Elaborazione delle predizioni e delle etichette per ogni task
            for task in ["gender", "bag", "hat"]:
                task_index = ["gender", "bag", "hat"].index(task)

                # Predizioni binarie (con soglia 0.5)
                preds = (torch.sigmoid(outputs[task]) > 0.5).int()

                # Filtra solo i campioni validi usando la maschera
                valid_preds = preds[masks[:, task_index]].cpu().numpy()
                valid_labels = labels[masks[:, task_index], task_index].cpu().numpy()

                # Salva predizioni ed etichette valide
                all_predictions[task].extend(valid_preds)
                all_labels[task].extend(valid_labels)

    # Calcolo delle metriche per ogni task
    metrics = {}
    output_dir = "./classification_andrea/confusion_matrices"  # Directory per salvare le matrici di confusione
    os.makedirs(output_dir, exist_ok=True)

    for task in ["gender", "bag", "hat"]:
        accuracy = accuracy_score(all_labels[task], all_predictions[task])
        precision = precision_score(all_labels[task], all_predictions[task], zero_division=0)
        recall = recall_score(all_labels[task], all_predictions[task], zero_division=0)
        f1 = f1_score(all_labels[task], all_predictions[task], zero_division=0)
        metrics[task] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        print(
            f"{task.capitalize()} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Calcola e salva la matrice di confusione
        cm = confusion_matrix(all_labels[task], all_predictions[task])
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["0", "1"], yticklabels=["0", "1"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for {task.capitalize()}")
        save_path = os.path.join(output_dir, f"confusion_matrix_{task}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix for {task} saved to {save_path}")
    return metrics




if __name__ == "__main__":
    # Configurazioni del test
    data_dir = './dataset'
    model_path = './classification_andrea/checkpoints/resnet50.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Caricamento del dataset di test
    test_dataset = TestDataset(data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Caricamento del modello
    model = PARMultiTaskNet(backbone='resnet50').to(device)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    # Test del modello
    print("\nTesting model...")
    metrics = test_model(model, test_loader, device)

    # Riepilogo delle metriche
    print("\nTest Summary:")
    for task, task_metrics in metrics.items():
        print(f"{task.capitalize()} Metrics: {task_metrics}")
