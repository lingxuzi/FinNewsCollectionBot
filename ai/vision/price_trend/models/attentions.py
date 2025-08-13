import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
def get_attention_module(channels, attention_mode='ca', **kwargs):
    if attention_mode == 'ca':
        return CA_Block(channels, **kwargs)
    elif attention_mode == 'mlca':
        return MLCA(channels, **kwargs)
    else:
        raise ValueError(f'Unknown attention mode: {attention_mode}')