import torch
import torch.nn as nn
from torchvision import models
def get_fsr(num_classes, ctx, kernel_size):
    """
    Definisce una rete neurale convoluzionale sequenziale in PyTorch,
    supportando il dispositivo specificato (`ctx`).

    Parametri:
    ----------
    num_classes : int
        Numero di classi per l'output (dimensione del livello finale).
    ctx : torch.device
        Dispositivo su cui allocare il modello (es. torch.device('cuda') o 'cpu').
    kernel_size : int o tuple
        Dimensione del kernel per il livello convoluzionale finale.

    Ritorna:
    --------
    nn.Sequential
        Modello sequenziale allocato sul dispositivo specificato.
    """
    # Definizione della rete sequenziale
    net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=kernel_size),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1024, num_classes)
    )

    # Inizializzazione dei pesi con Xavier
    for layer in net:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # Spostamento del modello sul dispositivo specificato
    return net.to(ctx)


def get_fatt(num_classes, stride, ctx):
    """
    Definisce una rete neurale convoluzionale in PyTorch,
    con supporto per il dispositivo specificato (`ctx`).

    Parametri:
    ----------
    num_classes : int
        Numero di classi per l'output (dimensione del livello finale).
    stride : int
        Passo per la convoluzione finale.
    ctx : torch.device
        Dispositivo su cui allocare il modello (es. torch.device('cuda') o 'cpu').

    Ritorna:
    --------
    nn.Sequential
        Modello sequenziale allocato sul dispositivo specificato.
    """
    # Definizione della rete sequenziale
    net = nn.Sequential(
        # Primo strato convoluzionale
        nn.Conv2d(in_channels=3, out_channels=512, kernel_size=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        # Secondo strato convoluzionale
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        # Secondo strato convoluzionale
        # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        # nn.BatchNorm2d(512),
        # nn.ReLU(),

        # Strato convoluzionale finale
        nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=stride)
    )

    # Inizializzazione Xavier per i pesi
    for layer in net:
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # Spostamento del modello sul dispositivo specificato
    return net.to(ctx)


def get_conv2D(num_classes, stride, ctx):
    """
    Definisce una semplice rete convoluzionale in PyTorch con attivazione sigmoid,
    e supporto per il dispositivo specificato (`ctx`).

    Parametri:
    ----------
    num_classes : int
        Numero di canali di output.
    stride : int
        Passo per la convoluzione.
    ctx : torch.device
        Dispositivo su cui allocare il modello (es. torch.device('cuda') o 'cpu').

    Ritorna:
    --------
    nn.Sequential
        Modello sequenziale allocato sul dispositivo specificato.
    """
    # Definizione della rete sequenziale
    net = nn.Sequential(
        # Convoluzione con kernel 1x1
        nn.Conv2d(in_channels=3, out_channels=num_classes, kernel_size=1, stride=stride),
        # Funzione di attivazione Sigmoid
        nn.Sigmoid()
    )

    # Inizializzazione Xavier per i pesi
    for layer in net:
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # Spostamento del modello sul dispositivo specificato
    return net.to(ctx)


def getResNet(num_classes, ctx, NoTraining=True):
    """
    Definisce un modello basato su ResNet-101 con estrazione di caratteristiche intermedie
    e supporto per il fine-tuning in PyTorch.

    Parametri:
    ----------
    num_classes : int
        Numero di classi per la classificazione finale.
    ctx : torch.device
        Dispositivo su cui allocare il modello (es. torch.device('cuda') o 'cpu').
    NoTraining : bool, opzionale
        Se True, disabilita i gradienti per il modello (default: True).

    Ritorna:
    --------
    feat_model : nn.Module
        Modello con estrazione di caratteristiche intermedie e output finale.
    """
    # 1. Carica ResNet-101 pre-addestrato
    resnet = models.resnet101(pretrained=True)
    resnet = resnet.to(ctx)  # Sposta il modello sul dispositivo specificato

    # 2. Modifica il livello di output per il numero di classi desiderato
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)  # Nuovo Fully Connected
    nn.init.xavier_normal_(resnet.fc.weight)  # Inizializza i pesi del nuovo livello
    resnet = resnet.to(ctx)  # Aggiorna sul dispositivo

    # 3. Seleziona i livelli convoluzionali come backbone
    features = nn.Sequential(*list(resnet.children())[:-2])  # Esclude il livello finale e AdaptiveAvgPool

    # 4. Definisci gli output intermedi da estrarre
    class FeatureExtractor(nn.Module):
        def __init__(self, backbone, fc, layers_to_extract):
            super().__init__()
            self.backbone = backbone  # Livelli convoluzionali
            self.fc = fc  # Livello Dense finale
            self.layers_to_extract = layers_to_extract  # Livelli intermedi da salvare

        def forward(self, x):
            features = []  # Lista per salvare le caratteristiche intermedie
            for idx, layer in enumerate(self.backbone.children()):
                x = layer(x)
                if idx in self.layers_to_extract:
                    features.append(x)
            output = self.fc(torch.flatten(x, 1))  # Appiattimento per il livello Dense
            features.append(output)  # Aggiunge l'output finale alle caratteristiche
            return features

    # 5. Livelli specifici da estrarre (ad esempio, ultimi strati convoluzionali e Dense)
    layers_to_extract = [6, 7]  # Stage convoluzionali specifici, possono essere adattati

    # 6. Combina backbone e strato Dense in un nuovo modello
    feat_model = FeatureExtractor(features, resnet.fc, layers_to_extract).to(ctx)

    # 7. Disabilita l'addestramento se richiesto
    if NoTraining:
        for param in feat_model.parameters():
            param.requires_grad = False

    return feat_model



def getDenseNet(num_classes, ctx):
    """
    Definisce un modello basato su DenseNet-201 con estrazione di caratteristiche intermedie
    e supporto per il fine-tuning in PyTorch.

    Parametri:
    ----------
    num_classes : int
        Numero di classi per la classificazione finale.
    ctx : torch.device
        Dispositivo su cui allocare il modello (es. torch.device('cuda') o 'cpu').

    Ritorna:
    --------
    feat_model : nn.Module
        Modello con estrazione di caratteristiche intermedie e output finale.
    """
    # 1. Carica DenseNet-201 pre-addestrato
    densenet = models.densenet201(pretrained=True)
    densenet = densenet.to(ctx)  # Sposta il modello sul dispositivo specificato

    # 2. Modifica il livello di output per il numero di classi desiderato
    densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)  # Nuovo Fully Connected
    nn.init.xavier_normal_(densenet.classifier.weight)  # Inizializza i pesi del nuovo livello
    densenet = densenet.to(ctx)  # Aggiorna sul dispositivo

    # 3. Seleziona i livelli convoluzionali come backbone
    features = densenet.features  # Struttura convoluzionale di DenseNet

    # 4. Definisci gli output intermedi da estrarre
    class FeatureExtractor(nn.Module):
        def __init__(self, backbone, classifier, layers_to_extract):
            super().__init__()
            self.backbone = backbone  # Livelli convoluzionali
            self.classifier = classifier  # Livello Dense finale
            self.layers_to_extract = layers_to_extract  # Livelli intermedi da salvare

        def forward(self, x):
            features = []  # Lista per salvare le caratteristiche intermedie
            for idx, layer in enumerate(self.backbone.children()):
                x = layer(x)
                if idx in self.layers_to_extract:
                    features.append(x)
            output = self.classifier(torch.flatten(x, 1))  # Appiattimento per il livello Dense
            features.append(output)  # Aggiunge l'output finale alle caratteristiche
            return features

    # 5. Livelli specifici da estrarre
    layers_to_extract = [6, 10]  # Indici dei livelli convoluzionali; possono essere personalizzati

    # 6. Combina backbone e strato Dense in un nuovo modello
    feat_model = FeatureExtractor(features, densenet.classifier, layers_to_extract).to(ctx)

    return feat_model