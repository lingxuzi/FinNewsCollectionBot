import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attentions import get_attention_module

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])
        self.conv2 = nn.Conv2d(in_channels // 2 * (len(kernel_sizes) + 1), out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        outputs = [x]
        for m in self.m:
            outputs.append(m(x))
        x = torch.cat(outputs, dim=1)
        x = self.conv2(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=5, stride=1, attention=True, attention_mode='ca'):
        super(ResidualBlock, self).__init__()
        self.attention = attention
        init_channels = out_channels // ratio
        # pointwise
        self.conv1 = nn.Conv2d(in_channels, init_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(init_channels)
        self.relu = nn.SiLU(inplace=True) #nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # depthwise
        self.conv2 = nn.Conv2d(init_channels, init_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=init_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(init_channels)
        self.ca = get_attention_module(init_channels, attention_mode)

        # pw-linear
        self.conv3 = nn.Conv2d(init_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.attention:
            out = self.ca(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(residual)
        return out
    
class StockChartNet(nn.Module):
    """
    一个集成了CBAM注意力机制的轻量级单一分支CNN模型。
    - 输入: (batch_size, 1, 60, 60) 的图像张量。
    - 输出: (batch_size, 1) 的预测收益率张量。
    """
    def __init__(self, pretrained=False, in_chans=1, attention_mode='ca'):
        super(StockChartNet, self).__init__()
        
        # --- 主干网络 ---
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.relu1 = nn.ReLU(inplace=True)

        stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        block2 = MixResidualBlock(16, 24, stride=2, attention=False)
        block3 = MixResidualBlock(24, 40, stride=2, attention=False)
        block4 = ResidualBlock(40, 128, stride=2, ratio=4, attention_mode=attention_mode)
        block5 = ResidualBlock(128, 256, stride=1, ratio=4, attention_mode=attention_mode)

        self.layers = nn.Sequential(
            stem,
            block2,
            block3,
            block4,
            block5
        )

        self.num_features = 256

    def forward_features(self, x):
        return self.layers(x)

class MixConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], stride=1):
        super(MixConv, self).__init__()

        channels_per_kernel = out_channels // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, channels_per_kernel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=channels_per_kernel, bias=False)
            for kernel_size in kernel_sizes
        ])
    
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)

class MixResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=3, kernel_sizes=[3, 5], stride=1, attention=True, attention_mode='ca'):
        super(MixResidualBlock, self).__init__()
        self.attention = attention
        init_channels = ((out_channels * ratio) // len(kernel_sizes)) * len(kernel_sizes)
        # pointwise
        self.conv1 = nn.Conv2d(in_channels, init_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(init_channels)
        self.relu = nn.SiLU(inplace=True) #nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # depthwise
        self.conv2 = MixConv(init_channels, init_channels, kernel_sizes, stride=stride)
        self.bn2 = nn.BatchNorm2d(init_channels)
        self.ca = get_attention_module(init_channels, attention_mode)

        # pw-linear
        self.conv3 = nn.Conv2d(init_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.attention:
            out = self.ca(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(residual)
        return out
    
class StockChartNetV2(nn.Module):
    """
    一个集成了CBAM注意力机制的轻量级单一分支CNN模型。
    - 输入: (batch_size, 1, 60, 60) 的图像张量。
    - 输出: (batch_size, 1) 的预测收益率张量。
    """
    def __init__(self, pretrained=False, in_chans=1, attention_mode='ca'):
        super(StockChartNetV2, self).__init__()
        
        # --- 主干网络 ---
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.relu1 = nn.ReLU(inplace=True)

        stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        block2 = MixResidualBlock(16, 24, stride=2, attention=False)
        block3 = MixResidualBlock(24, 40, stride=2, attention=False)
        block4 = MixResidualBlock(40, 80, stride=2, attention_mode=attention_mode)
        block5 = MixResidualBlock(80, 120, stride=2, attention_mode=attention_mode)

        self.layers = nn.Sequential(
            stem,
            block2,
            block3,
            block4,
            block5
        )

        self.num_features = 120

    def forward_features(self, x):
        return self.layers(x)