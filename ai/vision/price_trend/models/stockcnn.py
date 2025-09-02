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
        self.stride = stride  # 记录下采样倍数
        self.kernel_size = kernel_size
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
 
    def forward(self, x, mask):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.stride > 1:
            mask = F.max_pool2d(mask, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size//2)
        if self.attention:
            out = self.ca(out)
            out = out * mask
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(residual) * mask
        return out, mask
    
class StockChartNet(nn.Module):
    """
    一个集成了CBAM注意力机制的轻量级单一分支CNN模型。
    - 输入: (batch_size, 1, 60, 60) 的图像张量。
    - 输出: (batch_size, 1) 的预测收益率张量。
    """
    def __init__(self, pretrained=False, in_chans=1, attention_mode='ca'):
        super(StockChartNet, self).__init__()
        channels = [16, 32, 64, 128, 256]
        self.num_features = channels[-1]
        self.layers = self.build_conv_groups(in_chans, channels, 5, attention_mode)

    def build_conv_groups(self, in_chans, channels=[16, 32, 64, 128, 256], kernel_size = 5, attention_mode='ca'):
        self.stem = ResidualBlock(in_chans, channels[0], kernel_size=kernel_size, stride=2, attention=False)
        self.block2 = ResidualBlock(channels[0], channels[1], kernel_size=kernel_size, stride=2, attention=False)
        self.block3 = ResidualBlock(channels[1], channels[2], kernel_size=kernel_size, stride=2, attention=False)
        self.block4 = ResidualBlock(channels[2], channels[3], kernel_size=kernel_size, stride=1, attention_mode=attention_mode)
        self.block5 = ResidualBlock(channels[3], channels[4], kernel_size=kernel_size, stride=1, attention_mode=attention_mode)


    def forward_features(self, x):
        mask = (torch.sum(torch.abs(x), dim=1, keepdim=True) > 0.1).float()
        x, mask = self.stem(x, mask)
        x, mask = self.block2(x, mask)
        x, mask = self.block3(x, mask)
        x, mask = self.block4(x, mask)
        x, mask = self.block5(x, mask)
        return x
    
def _SplitChannels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class MixConv(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 5], stride=1):
        super(MixConv, self).__init__()

        self.split_channels = _SplitChannels(channels, len(kernel_sizes))
        self.convs = nn.ModuleList([
            nn.Conv2d(self.split_channels[i], self.split_channels[i], kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.split_channels[i], bias=False)
            for i, kernel_size in enumerate(kernel_sizes)
        ])
    
    def forward(self, x):
        x_split = torch.split(x, self.split_channels, dim=1)
        outputs = [conv(x_split[i]) for i, conv in enumerate(self.convs)]
        return torch.cat(outputs, dim=1)
    
class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_in_channels = _SplitChannels(in_channels, self.num_groups)
        self.split_out_channels = _SplitChannels(out_channels, self.num_groups)
        print(self.split_in_channels)
        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MixResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=3, kernel_sizes=[3, 5], stride=1, attention=True, attention_mode='ca'):
        super(MixResidualBlock, self).__init__()
        self.attention = attention
        init_channels = ((out_channels * ratio) // len(kernel_sizes)) * len(kernel_sizes)
        # pointwise
        self.conv1 = GroupedConv2d(in_channels, init_channels, kernel_size=[1], stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(init_channels)
        self.relu = nn.SiLU(inplace=True) #nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # depthwise
        self.conv2 = MixConv(init_channels, kernel_sizes, stride=stride)
        self.bn2 = nn.BatchNorm2d(init_channels)
        self.ca = get_attention_module(init_channels, attention_mode)

        # pw-linear
        self.conv3 = GroupedConv2d(init_channels, out_channels, kernel_size=[1], stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.residual_connection = (stride == 1 and in_channels == out_channels)
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )
 
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
        if self.residual_connection:
            out += residual
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