import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models as models
from .stockcnn import StockChartNet as stockchartnet
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
            

def weights_init(m):
    """
    为模型应用 Kaiming He 初始化。
    参数:
        m: PyTorch 模块 (nn.Module)
    """
    # classname = m.__class__.__name__
    # 对卷积层和全连接层使用 Kaiming Normal 初始化
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Kaiming Normal 初始化，专为ReLU设计
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        # 将偏置初始化为0
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # 对批量归一化层进行初始化
    elif isinstance(m, nn.BatchNorm2d) != -1:
        # 将权重(gamma)初始化为1，偏置(beta)初始化为0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def initialize(module: nn.Module):
    for name, m in module.named_modules():
        weights_init(m)

@register_model('stocknet')
class StockNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model = eval(f'{config["backbone"]}(pretrained=True, in_chans=1)')
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化
        
        self.trend_classifier = nn.Linear(self.model.num_features, config["trend_classes"])
        self.stock_classifier = nn.Linear(self.model.num_features, config["stock_classes"])
        self.industry_classifier = nn.Linear(self.model.num_features, config["industry_classes"])

        initialize(self)

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

    def gradcam_layer(self):
        return self.model.conv_head

