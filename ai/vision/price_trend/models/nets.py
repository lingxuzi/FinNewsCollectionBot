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


class FeatureFusedAttention(nn.Module):
    def __init__(self, attention_dim):
        super().__init__()
        self.v = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, vision_features, ts_features):
        # 2. 计算注意力权重
        attention_scores = self.v(torch.tanh(vision_features + ts_features))
        attention_weights = F.softmax(attention_scores, dim=1)

        # 3. 加权求和
        fused_features = attention_weights * ts_features

        return fused_features
    
class DropoutPredictionHead(nn.Module):
    def __init__(self, dropout=0.0, feature_dim=1280, classes=1, regression=False):
        super().__init__()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.classifier = nn.Linear(feature_dim, classes)
        if regression:
            weights_initialize(self.classifier)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.classifier(x)

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

        regression_output_size = config['ts_encoder']['embedding_dim']
        trend_output_size = 1280
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化
        self.fusion = FeatureFusedAttention(regression_output_size)
        
        self.trend_classifier = DropoutPredictionHead(feature_dim=trend_output_size, classes=config["trend_classes"], dropout=self.config['dropout'])
        self.trend_ts_classifier = DropoutPredictionHead(feature_dim=config['ts_encoder']['embedding_dim'], classes=config["trend_classes"], dropout=self.config['dropout'])
        self.trend_classifier_fused = DropoutPredictionHead(feature_dim=regression_output_size, classes=config["trend_classes"], dropout=self.config['dropout'])
        self.stock_classifier = DropoutPredictionHead(feature_dim=regression_output_size, classes=config["stock_classes"], dropout=self.config['dropout'])
        self.industry_classifier = DropoutPredictionHead(feature_dim=regression_output_size, classes=config["industry_classes"], dropout=self.config['dropout'])
        self.returns_regression = DropoutPredictionHead(feature_dim=regression_output_size, classes=1, dropout=self.config['dropout'], regression=True)

        if 'models.' not in config["backbone"]:
            initialize(self)
        
        orthogonal_init(self.ts_model)

    def export(self):
        self.eval()
        self.infer_mode = True

    def freeze_vision(self):
        for (name, param) in self.model.named_parameters():
            param.requires_grad = False
        
        for name, param in self.fusion.named_parameters():
            param.requires_grad = False

        self.last_conv.requires_grad_(False)
        self.trend_classifier.requires_grad_(False)
        self.trend_classifier_fused.requires_grad_(False)
        self.stock_classifier.requires_grad_(False)
        self.industry_classifier.requires_grad_(False)
        self.returns_regression.requires_grad_(False)
    
    def freeze_ts(self):
        for (name, param) in self.ts_model.named_parameters():
            param.requires_grad = False

        for name, param in self.fusion.named_parameters():
            param.requires_grad = False
            
        self.trend_ts_classifier.requires_grad_(False)
        self.trend_classifier_fused.requires_grad_(False)
        self.stock_classifier.requires_grad_(False)
        self.industry_classifier.requires_grad_(False)
        self.returns_regression.requires_grad_(False)

    def freeze_backbone(self):
        for (name, param) in self.model.named_parameters():
            param.requires_grad = False

        self.last_conv.requires_grad_(False)

        for (name, param) in self.ts_model.named_parameters():
            param.requires_grad = False

        self.last_conv.requires_grad_(False)
        self.trend_ts_classifier.requires_grad_(False)
        self.trend_classifier.requires_grad_(False)

    def forward(self, x, ts_seq, ctx_seq):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        # x = torch.cat([x, ts_emb.unsqueeze(2).unsqueeze(3)], dim=1)
        
        x = self.last_conv(x)
        x = self.hardswish(x)

        x = x.view(x.size(0), -1)

        if not self.infer_mode:
            trend_logits = self.trend_classifier(x)
        else:
            trend_logits = None
        
        if ts_seq is not None and ctx_seq is not None:
            ts_fused = self.ts_model((ts_seq, ctx_seq))
            if not self.infer_mode:
                ts_logits = self.trend_ts_classifier(ts_fused)
                stock_logits = self.stock_classifier(ts_fused)
                industry_logits = self.industry_classifier(ts_fused)
            else:
                ts_logits, stock_logits, industry_logits = None, None, None
            
            returns = self.returns_regression(ts_fused)

            ts_fused = self.fusion(x.detach(), ts_fused.detach())
            trend_logits_fused = self.trend_classifier_fused(ts_fused)
        else:
            ts_logits, trend_logits_fused, stock_logits, industry_logits, returns = None, None, None, None, None

        return trend_logits, ts_logits, trend_logits_fused, stock_logits, industry_logits, returns

    def gradcam_layer(self):
        return eval(f'self.model.{self.config["gradlayer"]}')

