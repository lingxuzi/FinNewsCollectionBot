import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialAttentionEnhanced(nn.Module):
    """增强版空间注意力：专门强化有效区域的空间定位"""
    def __init__(self, kernel_size=7):
        super(SpatialAttentionEnhanced, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 增加一个小权重，抑制背景区域的注意力值
        self.bg_suppress = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x):
        # 对特征图做全局平均和最大值池化（突出有效区域）
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        # 强化有效区域的注意力权重，抑制背景
        att_map = self.sigmoid(out) * (1 + self.bg_suppress)  # 有效区域权重提升
        return att_map


class ChannelAttention(nn.Module):
    """通道注意力：聚焦有效特征通道"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class CBAMEnhanced(nn.Module):
    """增强版CBAM：先空间注意力（锁定有效区域），再通道注意力（强化有效特征）"""
    def __init__(self, in_channels):
        super(CBAMEnhanced, self).__init__()
        self.spatial_att = SpatialAttentionEnhanced()  # 先空间定位
        self.channel_att = ChannelAttention(in_channels)  # 再通道筛选

    def forward(self, x):
        # 先通过空间注意力锁定有效区域
        x = x * self.spatial_att(x)
        # 再通过通道注意力强化有效特征
        x = x * self.channel_att(x)
        return x

class MLCA(nn.Module):
    def __init__(self, in_size,local_size=5,gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        # y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)  # 代码修正
        # print(y_global_transpose.size())
        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        # print(att_local.size())
        # print(att_global.size())
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])
        # print(att_all.size())
        x=x*att_all
        return x
    
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
 
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))


        mip = max(8, channel // reduction)
        
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=mip, kernel_size=1, stride=1)
 
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu = nn.Hardswish(inplace=True)
 
        self.F_h = nn.Conv2d(in_channels=mip, out_channels=channel, kernel_size=1, stride=1)
        self.F_w = nn.Conv2d(in_channels=mip, out_channels=channel, kernel_size=1, stride=1)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x, mask=None):
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

        att_map = A_h * A_w
        if mask is not None:
            att_map = att_map * mask
        out = x * att_map
 
        return out
    
class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
    
def get_attention_module(channels, attention_mode='ca', **kwargs):
    if attention_mode == 'ca':
        print('build with ca attention')
        return CA_Block(channels, **kwargs)
    elif attention_mode == 'mlca':
        print('build with mlca attention')
        return MLCA(channels, **kwargs)
    elif attention_mode == 'simam':
        print('build with simam attention')
        return SimAM(channels, **kwargs)
    elif attention_mode == 'cbam':
        print('build with cbam attention')
        return CBAMEnhanced(channels, **kwargs)
    else:
        raise ValueError(f'Unknown attention mode: {attention_mode}')
    
