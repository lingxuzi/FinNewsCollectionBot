import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm.models as models
from ai.vision.price_trend.models import register_model
from .stockcnn import StockChartNet as stockchartnet
from .stockcnn import StockChartNetV2 as stockchartnetv2
from .ts_encoder import TSEncoder

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

class AdditiveAttention(nn.Module):
    def __init__(self, vision_feature_dim, ts_feature_dim, attention_dim):
        super().__init__()
        self.W_v = nn.Linear(vision_feature_dim, attention_dim)
        self.W_t = nn.Linear(ts_feature_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, vision_features, ts_features):
        # 1. 计算注意力分数
        attention_scores = self.v(torch.tanh(self.W_v(vision_features).unsqueeze(1) + self.W_t(ts_features).unsqueeze(2))).squeeze(2)  # (B, 1)

        # 2. 计算注意力权重
        attention_weights = torch.softmax(attention_scores, dim=1)

        # 3. 加权求和
        attended_ts = ts_features * attention_weights
        # 或者:
        # attended_ts = torch.bmm(attention_weights.unsqueeze(1), ts_features.unsqueeze(1)).squeeze(1)

        # 4. 融合
        fused_features = torch.cat([vision_features, attended_ts], dim=1)

        return fused_features

@register_model('stocknet')
class StockNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.infer_mode = False
        
        self.model = eval(f'{config["backbone"]}(pretrained=True, in_chans=1, attention_mode="{config["attention_mode"]}")')
        self.ts_model = TSEncoder(config['ts_encoder'])

        self.last_conv = nn.Conv2d(
            in_channels=self.model.num_features,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.hardswish = nn.Hardswish()

        regression_output_size = 512 #1280 + config['ts_encoder']['embedding_dim']
        trend_output_size = 1280
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化
        self.fusion = AdditiveAttention(trend_output_size, config['ts_encoder']['embedding_dim'], regression_output_size)
        
        self.trend_classifier = nn.Linear(trend_output_size, config["trend_classes"])
        self.trend_ts_classifier = nn.Linear(config['ts_encoder']['embedding_dim'], config["trend_classes"])
        self.trend_classifier_fused = nn.Linear(regression_output_size, config["trend_classes"])
        self.stock_classifier = nn.Linear(regression_output_size, config["stock_classes"])
        self.industry_classifier = nn.Linear(regression_output_size, config["industry_classes"])
        self.returns_regression = nn.Linear(regression_output_size, 1)

        if 'models.' not in config["backbone"]:
            initialize(self.model)

    def export(self):
        self.eval()
        self.infer_mode = True

    def freeze_vision(self):
        for (name, param) in self.model.named_parameters():
            param.requires_grad = False
    
    def freeze_ts(self):
        for (name, param) in self.ts_model.named_parameters():
            param.requires_grad = False

    def forward(self, x, ts_seq, ctx_seq):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        # x = torch.cat([x, ts_emb.unsqueeze(2).unsqueeze(3)], dim=1)
        
        x = self.last_conv(x)
        x = self.hardswish(x)

        x = x.view(x.size(0), -1)

        if not self.infer_mode:
            if self.config['dropout'] > 0.:
                x = F.dropout(x, p=self.config['dropout'], training=self.training)

            trend_logits = self.trend_classifier(x)
        else:
            trend_logits = None
        
        if ts_seq is not None and ctx_seq is not None:
            ts_fused = self.ts_model((ts_seq, ctx_seq))
            if not self.infer_mode:
                if self.config['dropout'] > 0.:
                    ts_fused = F.dropout(ts_fused, p=self.config['dropout'], training=self.training)
                
                ts_logits = self.trend_ts_classifier(ts_fused)
            else:
                ts_logits = None

            ts_fused = self.fusion(x.detach(), ts_fused.detach())
            trend_logits_fused = self.trend_classifier_fused(ts_fused)
            if not self.infer_mode:
                stock_logits = self.stock_classifier(ts_fused)
                industry_logits = self.industry_classifier(ts_fused)
                returns = self.returns_regression(ts_fused)
            else:
                stock_logits, industry_logits, returns = None, None, None
        else:
            trend_logits_fused, stock_logits, industry_logits, returns = None, None, None, None

        return trend_logits, ts_logits, trend_logits_fused, stock_logits, industry_logits, returns

    def gradcam_layer(self):
        return eval(f'self.model.{self.config["gradlayer"]}')

