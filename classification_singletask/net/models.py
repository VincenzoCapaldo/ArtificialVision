import torch
from torch import nn
from torchvision import models
from utils import ClassBlock
from torchvision.models import ResNet50_Weights, ResNet34_Weights, DenseNet121_Weights

class Backbone_nFC(nn.Module):
    def __init__(self, class_num, model_name='resnet50_nfc'):
        super(Backbone_nFC, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num
        weights_mapping = {
            'resnet50': ResNet50_Weights.DEFAULT,
            'resnet34': ResNet34_Weights.DEFAULT,
            'densenet121': DenseNet121_Weights.DEFAULT,
        }
        # old code -> deprecated
        # model_ft = getattr(models, self.backbone_name)(pretrained=True)

        # Load model dynamically
        weights = weights_mapping.get(self.backbone_name, None)
        if not weights:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        model_ft = getattr(models, self.backbone_name)(weights=weights)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError
        
        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.num_ftrs, class_num=1, activ='sigmoid') )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        return pred_label

