import torch.nn as nn
import torchvision.models as models
from ultralytics.nn.modules import CBAM
import torch.nn.functional as F


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

        # Assicura che tutti i parametri abbiano requires_grad=True
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.params = list(self.backbone.parameters())

    def forward(self, x):
        return self.backbone(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # Modula i canali


class ClassificationHead(nn.Module):
    def __init__(self, num_classes=1, input_features=2048, attention=True):
        super(ClassificationHead, self).__init__()

        self.num_classes = num_classes
        self.input_features = input_features
        self.attention = attention

        if self.attention:
            self.attention = CBAM(self.input_features)
            self.se_block = SEBlock(self.input_features)  # Aggiungi SE Block

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(self.input_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        if self.attention:
            x = self.attention(x)
            x = self.se_block(x)  # SE Block dopo CBAM

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class PARMultiTaskNet(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, attention=True):
        super(PARMultiTaskNet, self).__init__()
        # defining backbone for feature extraction
        self.backbone = Backbone(name=backbone, pretrained=pretrained)

        self.gender_head = ClassificationHead(1, self.backbone.out_features, attention=attention)
        self.bag_head = ClassificationHead(1, self.backbone.out_features, attention=attention)
        self.hat_head = ClassificationHead(1, self.backbone.out_features, attention=attention)

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
