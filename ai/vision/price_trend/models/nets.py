import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models as models
from timm.layers import create_classifier
from ai.vision.price_trend.models import register_model

@register_model('stocknet')
class StockNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model = eval(f'models.{config["backbone"]}(pretrained=True, in_chans=1)')
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.trend_classifier = nn.Linear(self.model.num_features, config["trend_classes"])
        self.stock_classifier = nn.Linear(self.model.num_features, config["stock_classes"])
        self.industry_classifier = nn.Linear(self.model.num_features, config["industry_classes"])

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        # flatten
        x = x.view(x.size(0), -1)
        if self.config['dropout'] > 0.:
            x = F.dropout(x, p=self.config['dropout'], training=self.training)
        trend_logits = self.trend_classifier(x)
        stock_logits = self.stock_classifier(x)
        industry_logits = self.industry_classifier(x)
        return trend_logits, stock_logits, industry_logits

