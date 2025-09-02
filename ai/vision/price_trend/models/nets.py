import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm.models as models
from ai.vision.price_trend.models import register_model
from .stockcnn import StockChartNet as stockchartnet
from .stockcnn import StockChartNetV2 as stockchartnetv2
from .fuse import get_fusing_layer
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
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def orthogonal_init(model):
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            if len(param.shape) >= 2: # 仅初始化权重矩阵
                torch.nn.init.orthogonal_(param)

def weights_initialize(module):
    print("🧠 Initializing prediction head for faster convergence...")
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
        print(f"   -> Linear layer {module} has been zero-initialized.")
    else:
        print(f"   -> Module {type(module)} is not a Linear layer, skipping zero-initialization.")

class DropoutPredictionHead(nn.Module):
    def __init__(self, dropout=0.0, feature_dim=1280, classes=1, dropout_samples=8, regression=False):
        super().__init__()
        if dropout > 0.0:
            self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(dropout_samples)])
        else:
            self.dropout = None

        self.classifier = nn.Linear(feature_dim, classes)
        if regression:
            weights_initialize(self.classifier)

    def forward(self, x):
        if self.training:
            outputs = []
            if self.dropout is not None:
                for dropout in self.dropout:
                    x = dropout(x)
                    outputs.append(self.classifier(x))
            else:
                outputs.append(self.classifier(x))
            return outputs
        else:
            return self.classifier(x)

@register_model('stocknet')
class StockNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.infer_mode = False
        self.config = config

    def build_vision(self):
        self.model = eval(f'{self.config["backbone"]}(pretrained=True, in_chans=1, attention_mode="{self.config["attention_mode"]}")')
        self.last_conv = nn.Conv2d(
            in_channels=self.model.num_features,
            out_channels=self.config['embedding_dim'],  
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.hardswish = nn.SiLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化
        self.trend_classifier = DropoutPredictionHead(feature_dim=self.config['embedding_dim'], classes=self.config["trend_classes"], dropout=self.config['dropout'])
        if 'models.' not in self.config["backbone"]:
            initialize(self.model)
    
    def build_ts(self):
        self.ts_model = TSEncoder(self.config['ts_encoder'])
        self.trend_ts_classifier = DropoutPredictionHead(feature_dim=self.config['embedding_dim'], classes=self.config["trend_classes"], dropout=self.config['dropout'])
        orthogonal_init(self.ts_model)

    def build_fusion(self):
        self.fusion = get_fusing_layer(self.config['fused_method'], fused_dim=self.config['embedding_dim'], hidden_dim=self.config['embedding_dim'] // 2)

        self.trend_classifier_fused = DropoutPredictionHead(feature_dim=self.config['embedding_dim'], classes=self.config["trend_classes"], dropout=self.config['dropout'])
        self.stock_classifier = DropoutPredictionHead(feature_dim=self.config['embedding_dim'], classes=self.config["stock_classes"], dropout=self.config['dropout'])
        self.industry_classifier = DropoutPredictionHead(feature_dim=self.config['embedding_dim'], classes=self.config["industry_classes"], dropout=self.config['dropout'])
        self.returns_regression = DropoutPredictionHead(feature_dim=self.config['embedding_dim'], classes=1, dropout=self.config['dropout'], regression=True)

    def export(self):
        self.eval()
        self.infer_mode = True

    def forward(self, x, ts_seq, ctx_seq):
        if self.infer_mode:
            return self.inference(x, ts_seq, ctx_seq)

    def __vision_features(self, x):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        # x = torch.cat([x, ts_emb.unsqueeze(2).unsqueeze(3)], dim=1)
        
        x = self.last_conv(x)
        x = self.hardswish(x)

        x = x.view(x.size(0), -1)
        return x
    
    def __classify_vision(self, vision_features):
        trend_logits = self.trend_classifier(vision_features)
        return trend_logits
    
    def vision_logits(self, x):
        vision_features = self.__vision_features(x)
        trend_logits = self.__classify_vision(vision_features)
        return {
            'vision_logits': trend_logits
        }

    def __ts_features(self, ts_seq, ctx_seq):
        ts_features = self.ts_model((ts_seq, ctx_seq))
        return ts_features
    
    def __classify_ts(self, ts_features):
        trend_logits = self.trend_ts_classifier(ts_features)
        return trend_logits
    
    def ts_logits(self, ts_seq, ctx_seq):
        ts_features = self.__ts_features(ts_seq, ctx_seq)
        trend_logits = self.__classify_ts(ts_features)
        return {
            'ts_logits': trend_logits
        }
    
    def fuse_logits(self, x, ts_seq, ctx_seq):
        with torch.no_grad():
            vision_features = self.__vision_features(x)
            ts_features = self.__ts_features(ts_seq, ctx_seq)
        fused_features = self.fusion(vision_features.detach(), ts_features.detach())
        # fused_features = F.dropout(fused_features, p=self.config['dropout'], training=self.training)
        trend_logits_fused = self.trend_classifier_fused(fused_features)
        stock_logits = self.stock_classifier(fused_features)
        industry_logits = self.industry_classifier(fused_features)
        returns = self.returns_regression(fused_features)
        return {
            'fused_trend_logits': trend_logits_fused,
            'stock_logits': stock_logits,
            'industry_logits': industry_logits,
            'returns': returns
        }
    
    def inference(self, x, ts_seq, ctx_seq):
        with torch.no_grad():
            vision_features = self.__vision_features(x)
            ts_features = self.__ts_features(ts_seq, ctx_seq)
            fused_features = self.fusion(vision_features.detach(), ts_features.detach())
            trend_logits_fused = self.trend_classifier_fused(fused_features)
            returns = self.returns_regression(fused_features)
            vision_logits = self.__classify_vision(vision_features)
            ts_logits = self.__classify_ts(ts_features)

            return {
                'fused_trend_logits': trend_logits_fused,
                'vision_logits': vision_logits,
                'ts_logits': ts_logits,
                'returns': returns
            }
    
    def all_logits(self, x, ts_seq, ctx_seq):
        vision_features = self.__vision_features(x)
        ts_features = self.__ts_features(ts_seq, ctx_seq)

        vision_logits = self.__classify_vision(vision_features)
        ts_logits = self.__classify_ts(ts_features)

        fused_features = self.fusion(vision_features.detach(), ts_features.detach())
        trend_logits_fused = self.trend_classifier_fused(fused_features)
        # stock_logits = self.stock_classifier(fused_features)
        # industry_logits = self.industry_classifier(fused_features)
        returns = self.returns_regression(fused_features)
        return {
            'fused_trend_logits': trend_logits_fused,
            'vision_logits': vision_logits,
            'ts_logits': ts_logits,
            'returns': returns
        }

    def gradcam_layer(self):
        return eval(f'self.model.{self.config["gradlayer"]}')

