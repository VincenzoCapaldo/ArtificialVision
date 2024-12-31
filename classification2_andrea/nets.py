import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights, ConvNeXt_Base_Weights
from ultralytics.nn.modules import CBAM


class Backbone(nn.Module):
    def __init__(self, name="resnet50"):
        super(Backbone, self).__init__()

        if name == "densenet":
            backbone = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            self.out_features = backbone.classifier.in_features  # Numero di canali in uscita
            # Rimuovi il global average pooling e il layer classifier
            self.backbone = nn.Sequential(*list(backbone.features.children()))
        elif name == "resnet18":
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.out_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "resnet50":
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "convnext_base":
            backbone = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            self.out_features = 1024
            self.backbone = backbone.features

        # Assicura che tutti i parametri abbiano requires_grad=True
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.params = list(self.backbone.parameters())

    def forward(self, x):
        return self.backbone(x)


class ClassificationHead(nn.Module):
    """
    ClassificationHead è una classe che combina attenzione, pooling adattivo,
    e una serie di layer fully connected per la classificazione.
    Ideale per compiti di classificazione binaria o multiclasse, utilizzando
    feature estratte da un backbone come ResNet50.
    """

    def __init__(self, num_classes=1, input_features=2048):
        """
        Args:
            num_classes (int): Numero di classi da classificare. Se 1, esegue classificazione binaria.
            input_features (int): Numero di feature in input, di default 2048 (output di ResNet50).
        """
        super(ClassificationHead, self).__init__()

        self.num_classes = num_classes
        self.input_features = input_features

        # Attention mechanism: utilizza CBAM per applicare attenzione sui canali e spazio
        self.attention = CBAM(self.input_features)

        # Pooling adattivo per ridurre le dimensioni spaziali a 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Serie di layer fully connected per il processo decisionale
        self.fc1 = nn.Linear(self.input_features, 512)  # first dense layer
        self.fc2 = nn.Linear(512, 256)  # second dense layer
        self.fc3 = nn.Linear(256, 128)  # third dense layer
        self.fc4 = nn.Linear(128, num_classes)  # fourth dense layer

        # Dropout per prevenire overfitting durante l'addestramento
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass attraverso il modello.

        Args:
            x (torch.Tensor): Input tensor, tipicamente feature estratte dal backbone.

        Returns:
            torch.Tensor: Output dei logit per classificazione multiclasse,
                          o probabilità per classificazione binaria.
        """
        # Applica il modulo di attenzione

        x = self.attention(x)

        # Riduce la dimensione spaziale a 1x1 tramite pooling adattivo
        x = self.avgpool(x)

        # Appiattisce il tensore da 4D (B x C x 1 x 1) a 2D (B x C)
        x = x.view(x.size(0), -1)

        # Passa attraverso i layer fully connected con ReLU e dropout
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  # Dropout applicato prima del layer finale
        x = self.fc4(x)  # Layer finale

        return x


class PARMultiTaskNet(nn.Module):
    def __init__(self, backbone="resnet50"):
        super(PARMultiTaskNet, self).__init__()
        # defining backbone for feature extraction
        self.backbone = Backbone(name=backbone)
        self.gender_head = ClassificationHead(1, self.backbone.out_features)
        self.bag_head = ClassificationHead(1, self.backbone.out_features)
        self.hat_head = ClassificationHead(1, self.backbone.out_features)

        self.gender_loss = nn.BCEWithLogitsLoss()
        self.bag_loss = nn.BCEWithLogitsLoss()
        self.hat_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # getting extracted features
        features = self.backbone(x)

        gender_output = self.gender_head(features)
        bag_output = self.bag_head(features)
        hat_output = self.hat_head(features)

        return {
            "gender": gender_output,
            "bag": bag_output,
            "hat": hat_output
        }

    def compute_masked_loss(self, y_pred, y_true):
        # La masked loss calcola la media delle loss solo dei campioni
        # validi, cioè con la label != -1.
        # labels order: gender, bag, hat

        y_true = torch.stack(y_true)  # Ora [batch_size, 3]
        # Estrazione di etichette per ciascun task
        gender_label = y_true[:, 0].unsqueeze(1).squeeze(-1)  # Primo task: Gender
        bag_label = y_true[:, 1].unsqueeze(1).squeeze(-1)  # Secondo task: Bag
        hat_label = y_true[:, 2].unsqueeze(1).squeeze(-1)  # Terzo task: Hat
        gender_pred = y_pred["gender"]
        bag_pred = y_pred["bag"]
        hat_pred = y_pred["hat"]

        # ---- GENDER ----
        # Crea la maschera per i valori validi
        mask_gender = (gender_label >= 0).float()
        # Espandi le dimensioni delle etichette per la loss (es. [batch_size] -> [batch_size, 1])
        gender_label = gender_label.unsqueeze(1).float()
        # Calcola la loss per tutti gli esempi
        loss_gender_all = self.gender_loss(gender_pred, gender_label)
        # Applica la maschera e calcola la media solo per i valori validi
        loss_gender = (loss_gender_all * mask_gender).sum() / (mask_gender.sum() + 1e-8)

        # ---- BAG ----
        # Crea la maschera per i valori validi
        mask_bag = (bag_label >= 0).float()
        # Espandi le dimensioni delle etichette per la loss
        bag_label = bag_label.unsqueeze(1).float()
        # Calcola la loss per tutti gli esempi
        loss_bag_all = self.bag_loss(bag_pred, bag_label)
        # Applica la maschera e calcola la media solo per i valori validi
        loss_bag = (loss_bag_all * mask_bag).sum() / (mask_bag.sum() + 1e-8)

        # ---- HAT ----
        # Crea la maschera per i valori validi
        mask_hat = (hat_label >= 0).float()
        # Espandi le dimensioni delle etichette per la loss
        hat_label = hat_label.unsqueeze(1).float()
        # Calcola la loss per tutti gli esempi
        loss_hat_all = self.hat_loss(hat_pred, hat_label)
        # Applica la maschera e calcola la media solo per i valori validi
        loss_hat = (loss_hat_all * mask_hat).sum() / (mask_hat.sum() + 1e-8)

        return [loss_gender, loss_bag, loss_hat]

