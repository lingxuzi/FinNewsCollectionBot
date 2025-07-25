import torchvision.models as models
import torch
import torch.nn as nn

from ai.vision.price_trend.models import register_model

@register_model('effnet')
class EffNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True, num_classes=num_classes)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride = (2, 2), padding = (1,1), bias=False)

    def forward(self, x):
        return self.model(x)