import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
 
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.bn1 = nn.BatchNorm2d(channel//reduction)
        self.relu = nn.Hardswish(inplace=True)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
 
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        x_cat = torch.cat([x_h, x_w], dim=2)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.relu(x_cat)
 
        x_h = x_cat[:, :, :h, :]
        x_w = x_cat[:, :, h:, :].permute(0, 1, 3, 2)
 
        A_h = self.sigmoid_h(self.F_h(x_h))
        A_w = self.sigmoid_w(self.F_w(x_w))
 
        out = x * A_h * A_w
 
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, stride=1, attention=True):
        super(ResidualBlock, self).__init__()
        self.attention = attention
        init_channels = math.ceil(out_channels / ratio) 
        self.conv1 = nn.Conv2d(in_channels, init_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(init_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = CA_Block(out_channels, reduction=8)
 
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
        if self.attention:
            out = self.ca(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
    
class StockChartNet(nn.Module):
    """
    一个集成了CBAM注意力机制的轻量级单一分支CNN模型。
    - 输入: (batch_size, 1, 60, 60) 的图像张量。
    - 输出: (batch_size, 1) 的预测收益率张量。
    """
    def __init__(self, pretrained=False, in_chans=1):
        super(StockChartNet, self).__init__()
        
        # --- 主干网络 ---
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.relu1 = nn.ReLU(inplace=True)

        stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )

        block2 = ResidualBlock(16, 32, stride=2, attention=False)
        block3 = ResidualBlock(32, 64, stride=2)
        block4 = ResidualBlock(64, 128, stride=2)

        self.layers = nn.Sequential(
            stem,
            block2,
            block3,
            block4
        )

        self.num_features = 128

    def forward_features(self, x):
        return self.layers(x)
    

