import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TestDataset
from classification_andrea.nets import PARMultiTaskNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Funzione per testare il modello
def test_model(model, dataloader, device):
    model.eval()

    all_labels = {"gender": [], "bag": [], "hat": []}
    all_predictions = {"gender": [], "bag": [], "hat": []}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Test..."):
            images, labels = images.to(device), labels.to(device)
            masks = labels >= 0

            outputs = model(images)

            # Estrazione delle predizioni
            for task in ["gender", "bag", "hat"]:
                task_index = ["gender", "bag", "hat"].index(task)
                preds = (torch.sigmoid(outputs[task]) > 0.5).int()
                all_predictions[task].extend(preds[masks[:, task_index]].cpu().numpy())
                all_labels[task].extend(labels[masks[:, task_index], task_index].cpu().numpy())

    # Calcolo delle metriche per ogni task
    metrics = {}
    for task in ["gender", "bag", "hat"]:
        accuracy = accuracy_score(all_labels[task], all_predictions[task])
        precision = precision_score(all_labels[task], all_predictions[task], zero_division=0)
        recall = recall_score(all_labels[task], all_predictions[task], zero_division=0)
        f1 = f1_score(all_labels[task], all_predictions[task], zero_division=0)
        metrics[task] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        print(
            f"{task.capitalize()} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return metrics


if __name__ == "__main__":
    # Configurazioni del test
    data_dir = './dataset'
    checkpoint_path = './classification_andrea/models/resnet50 con adam e loss pesata.pth'
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Caricamento del dataset di test
    test_dataset = TestDataset(data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Caricamento del modello
    model = PARMultiTaskNet(backbone_name='resnet50', pretrained=False).to(device)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    # Test del modello
    print("\nTesting model...")
    metrics = test_model(model, test_loader, device)

    # Riepilogo delle metriche
    print("\nTest Summary:")
    for task, task_metrics in metrics.items():
        print(f"{task.capitalize()} Metrics: {task_metrics}")
