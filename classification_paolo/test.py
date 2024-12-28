import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import CustomDataset
from nets import ClassificationModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda o cpu')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone name: resnet50 o resnet18')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--type_classifier', type=str, default="gender", help='type of classifier')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Usando il device: {device}")

    # Percorsi dei dati
    image_dir = "dataset/test_set"
    label_file = "dataset/test_set.txt"

    # Caricamento del dataset e DataLoader
    dataset = CustomDataset(image_dir, label_file, type=args.type_classifier, test=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=args.num_workers)

    # Modello
    print(f"Caricamento modello migliore per il riconoscimento di {args.type_classifier}")
    model = ClassificationModel(args.backbone)
    model = model.to(device)
    model_path = f'./classification_paolo/models/best_{args.type_classifier}_model.pth'  # Sostituisci con il percorso del modello salvato
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Imposta il modello in modalità valutazione

    # Calcolo delle metriche
    accuracy, precision, recall, f1 = calculate_metrics(model, dataloader)
    print(f"Accuratezza: {accuracy:.4f}")
    print(f"Precisione: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

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
    main()
