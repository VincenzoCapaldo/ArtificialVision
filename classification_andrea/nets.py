import torch
import torch.nn as nn
import torchvision.models as models


# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         """
#         CBAM: Channel and Spatial Attention Module
#         :param in_channels: Numero di canali in ingresso
#         :param reduction_ratio: Fattore di riduzione per l'attenzione canale
#         :param kernel_size: Dimensione del kernel per l'attenzione spaziale
#         """
#         super(CBAM, self).__init__()
#
#         # Channel Attention Module
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.AdaptiveMaxPool2d(1),
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )
#
#         # Spatial Attention Module
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # Channel Attention
#         avg_pool = self.channel_attention[0](x)
#         max_pool = self.channel_attention[1](x)
#         channel_attn = self.channel_attention[2:](avg_pool + max_pool)
#         x = x * channel_attn
#
#         # Spatial Attention
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_attn = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
#         x = x * spatial_attn
#
#         return x


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


class PARMultiTaskNet(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True, num_classes=1):
        """
        Inizializza la rete multitask per PAR.
        :param backbone_name: Nome della backbone da utilizzare (es. 'resnet50')
        :param pretrained: Carica pesi pre-addestrati (True/False)
        :param num_classes: Numero di classi per ogni task (default 1 per classificazione binaria)
        """
        super(PARMultiTaskNet, self).__init__()

        # Dynamically load the specified backbone with or without pretrained weights
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")

        # Rimuove il classificatore fully connected (fc)
        self.backbone_features = nn.Sequential(*list(self.backbone.children())[:-2])

        # Testa comune per estrarre feature condivise (flattened features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Moduli di attenzione specifici per task
        self.gender_attention = TaskSpecificAttention(self.backbone.fc.in_features)
        self.bag_attention = TaskSpecificAttention(self.backbone.fc.in_features)
        self.hat_attention = TaskSpecificAttention(self.backbone.fc.in_features)
        # self.gender_attention = CBAM(self.backbone.fc.in_features)
        # self.bag_attention = CBAM(self.backbone.fc.in_features)
        # self.hat_attention = CBAM(self.backbone.fc.in_features)

        # Tre teste di classificazione
        self.gender_head = self._create_classification_head(self.backbone.fc.in_features, num_classes)
        self.bag_head = self._create_classification_head(self.backbone.fc.in_features, num_classes)
        self.hat_head = self._create_classification_head(self.backbone.fc.in_features, num_classes)

    def _create_classification_head(self, in_features, num_classes):
        """
        Crea una testa di classificazione.
        :param in_features: Numero di feature in ingresso
        :param num_classes: Numero di classi in uscita (default 1 per classificazione binaria)
        :return: Modulo nn.Sequential per classificazione
        """
        return nn.Sequential(
            nn.Linear(in_features, 512),  # Maggiore capacità per task complessi
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Passa i dati attraverso la rete.
        :param x: Input tensor
        :return: Output delle tre teste (gender, bag, hat)
        """
        # Passa attraverso la backbone
        features = self.backbone_features(x)  # [batch_size, channels, h, w]
        features = self.avgpool(features)  # [batch_size, channels, 1, 1]
        features = torch.flatten(features, 1)  # [batch_size, channels]

        # Applica moduli di attenzione specifici per task
        gender_features = self.gender_attention(features)
        bag_features = self.bag_attention(features)
        hat_features = self.hat_attention(features)

        # # Applica CBAM come attenzione specifica per task
        # gender_features = self.gender_attention(features.unsqueeze(-1).unsqueeze(-1))  # Aggiunge dimensioni spaziali
        # bag_features = self.bag_attention(features.unsqueeze(-1).unsqueeze(-1))
        # hat_features = self.hat_attention(features.unsqueeze(-1).unsqueeze(-1))
        #
        # gender_features = gender_features.squeeze(-1).squeeze(-1)  # Rimuove dimensioni spaziali
        # bag_features = bag_features.squeeze(-1).squeeze(-1)
        # hat_features = hat_features.squeeze(-1).squeeze(-1)

        # Calcola output delle tre teste
        gender_output = self.gender_head(gender_features)
        bag_output = self.bag_head(bag_features)
        hat_output = self.hat_head(hat_features)

        return {
            "gender": gender_output,
            "bag": bag_output,
            "hat": hat_output
        }


if __name__ == "__main__":
    # Testing with pretrained ResNet50
    model = PARMultiTaskNet(backbone_name="resnet50", pretrained=True)
    dummy_input = torch.randn(8, 3, 288, 144)
    outputs = model(dummy_input)
    print("ResNet50 (Pretrained) Outputs:")
    print("Gender output shape:", outputs["gender"].shape)
    print("Bag output shape:", outputs["bag"].shape)
    print("Hat output shape:", outputs["hat"].shape)

    # Testing with non-pretrained ResNet18
    model = PARMultiTaskNet(backbone_name="resnet18", pretrained=False)
    outputs = model(dummy_input)
    print("\nResNet18 (Non-Pretrained) Outputs:")
    print("Gender output shape:", outputs["gender"].shape)
    print("Bag output shape:", outputs["bag"].shape)
    print("Hat output shape:", outputs["hat"].shape)
