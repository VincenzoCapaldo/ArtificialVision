import torch.nn as nn
from torchvision import models

def create_classification_head(input_features):
    """Crea una testa di classificazione con layer fully connected personalizzati."""
    return nn.Sequential(
        nn.Linear(input_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )

class ClassificationModel(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(ClassificationModel, self).__init__()

        # Dynamically load the specified backbone with or without pretrained weights
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Congela tutti i parametri del backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Sblocca l'ultimo blocco convoluzionale
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Head: layer fully connected personalizzati
        num_ftrs = self.backbone.fc.in_features
        self.head = create_classification_head(num_ftrs)

        # Sostituisci il layer finale del backbone con la testa personalizzata
        self.backbone.fc = self.head

    def forward(self, x):
        return self.backbone(x)
