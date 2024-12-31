import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics.nn.modules import CBAM


class Backbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super(Backbone, self).__init__()

        if name == "densenet":
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            self.out_features = backbone.classifier.in_features  # Numero di canali in uscita
            # Rimuovi il global average pooling e il layer classifier
            self.backbone = nn.Sequential(*list(backbone.features.children()))
        elif name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.out_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.out_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "convnext_base":
            backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
            self.out_features = 1024
            self.backbone = backbone.features

        # Assicura che tutti i parametri abbiano requires_grad=True
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.params = list(self.backbone.parameters())

    def forward(self, x):
        return self.backbone(x)


class TaskSpecificAttention(nn.Module):
    def __init__(self, in_features, num_heads=4):
        """
        Modulo di attenzione specifica per task.
        :param in_features: Dimensione delle feature in ingresso
        :param num_heads: Numero di teste di attenzione (default: 4)
        """
        super(TaskSpecificAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads)
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        Applica il modulo di attenzione specifica.
        :param x: Input tensor [batch_size, in_features]
        :return: Tensor processato con attenzione [batch_size, in_features]
        """
        # Aggiungi una dimensione per rappresentare la sequenza (seq_len=1)
        x = x.unsqueeze(1)  # [batch_size, 1, in_features]
        # Calcola l'attenzione
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        # Applica la normalizzazione layer-wise con residual connection
        x = self.norm(self.dropout(attn_output) + x)  # Residual connection
        return x.squeeze(1)  # Rimuovi la dimensione della sequenza


class ClassificationHeadCbam(nn.Module):
    """
    ClassificationHeadCbam è una classe che combina attenzione CBAM, pooling adattivo,
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
        super(ClassificationHeadCbam, self).__init__()

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
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        # x = self.dropout(x)
        x = nn.ReLU(self.fc3(x))
        x = self.dropout(x)  # Dropout applicato prima del layer finale
        x = self.fc4(x)  # Layer finale

        return x


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

        # Testa comune per estrarre feature condivise (flattened features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Moduli di attenzione specifici per task
        self.attention = TaskSpecificAttention(input_features)

        # Serie di layer fully connected per il processo decisionale
        self.fc1 = nn.Linear(self.input_features, 512)  # first dense layer
        self.fc2 = nn.Linear(512, 256)  # second dense layer
        self.fc3 = nn.Linear(256, num_classes)  # third dense layer
        # self.fc4 = nn.Linear(128, num_classes)  # fourth dense layer

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
        x = self.avgpool(x)  # [batch_size, channels, 1, 1]
        x = torch.flatten(x, 1)  # [batch_size, channels]

        # Applica moduli di attenzione specifici per task
        x = self.attention(x)

        # Passa attraverso i layer fully connected con ReLU e dropout
        x = nn.ReLU(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU(self.fc2(x))
        x = self.dropout(x)  # Dropout applicato prima del layer finale
        x = self.fc3(x)
        #x = self.fc4(x)  # Layer finale


class PARMultiTaskNet(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, cbam=False):
        super(PARMultiTaskNet, self).__init__()
        # defining backbone for feature extraction
        self.backbone = Backbone(name=backbone, pretrained=pretrained)
        if cbam:
            self.gender_head = ClassificationHeadCbam(1, self.backbone.out_features)
            self.bag_head = ClassificationHeadCbam(1, self.backbone.out_features)
            self.hat_head = ClassificationHeadCbam(1, self.backbone.out_features)
        else:
            self.gender_head = ClassificationHead(1, self.backbone.out_features)
            self.bag_head = ClassificationHead(1, self.backbone.out_features)
            self.hat_head = ClassificationHead(1, self.backbone.out_features)

        self.gender_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bag_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.hat_loss = nn.BCEWithLogitsLoss(reduction='none')

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
