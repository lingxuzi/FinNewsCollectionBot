import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm.models as models
from .stockcnn import StockChartNet as stockchartnet
from .stockcnn import StockChartNetV2 as stockchartnetv2
from ai.vision.price_trend.models import register_model

class cnn20d(nn.Module):
    def __init__(self, pretrained=False, in_chans=1):
        super().__init__()
        block1 = nn.Sequential(nn.Conv2d(in_chans, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(3, 1)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                    nn.MaxPool2d((2, 1)))
        block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1)),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                    nn.MaxPool2d((2, 1)))
        block3 = nn.Sequential(nn.Conv2d(128, 256, (5, 3), padding=(3, 1)),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                    nn.MaxPool2d((2, 1)))

        self.layers = nn.Sequential(block1, block2, block3)
        self.num_features = 256

    def forward_features(self, x):
        return self.layers(x)
            

def initialize(module: nn.Module):
    for m in module.modules():
        #print(m)
        if isinstance(m, nn.Conv2d):
            #print(m.weight.size())
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

@register_model('stocknet')
class StockNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model = eval(f'{config["backbone"]}(pretrained=True, in_chans=1)')


        self.last_conv = nn.Conv2d(
            in_channels=self.model.num_features,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.hardswish = nn.Hardswish()

        output_size = 1280
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化


        # Embedding layers for stock and industry
        self.stock_embedding = nn.Embedding(config["stock_classes"], 64)
        self.industry_embedding = nn.Embedding(config["industry_classes"], 32)

        # Define the size of the combined embedding vector
        combined_embedding_size = 64 + 32

        # Define a linear layer to project the combined embedding vector to a specific size
        self.embedding_projection = nn.Linear(combined_embedding_size, 32)
        
        self.trend_classifier = nn.Linear(output_size, config["trend_classes"])

        if 'models.' not in config["backbone"]:
            initialize(self.model)

    def forward(self, x, stock, industry):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        # flatten
        x = x.view(x.size(0), -1)

        # Get stock and industry embeddings
        stock_embedding = self.stock_embedding(stock)
        industry_embedding = self.industry_embedding(industry)

        # Concatenate stock and industry embeddings
        combined_embedding = torch.cat((stock_embedding, industry_embedding), dim=1)

        # Project the combined embedding vector to a specific size
        projected_embedding = self.embedding_projection(combined_embedding)

        # Concatenate the CNN features with the projected embedding
        x = torch.cat((x, projected_embedding), dim=1)

        if self.config['dropout'] > 0.:
            x = F.dropout(x, p=self.config['dropout'], training=self.training)

        trend_logits = self.trend_classifier(x)
        return trend_logits

    def gradcam_layer(self):
        return eval(f'self.model.{self.config["gradlayer"]}')

