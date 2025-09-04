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
    
class LightweightSelfAttention(nn.Module):
    """
    轻量级自注意力模块
    特点：
    1. 参数共享机制，使用单个投影层生成QKV基础特征
    2. 简化的维度转换，减少中间计算量
    3. 可选的稀疏注意力模式，进一步降低计算复杂度
    4. 精简的残差连接设计
    """
    def __init__(self, 
                 dim, 
                 head_dim=32,  # 每个注意力头的维度
                 dropout=0.1,
                 sparse=False,  # 是否启用稀疏注意力
                 sparse_window=5):  # 稀疏注意力的窗口大小
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.sparse = sparse
        self.sparse_window = sparse_window
        
        # 计算头数（确保能整除）
        self.num_heads = dim // head_dim
        assert self.num_heads * head_dim == dim, "dim必须是head_dim的整数倍"
        
        # 单个投影层替代三个QKV投影层，大幅减少参数
        self.qkv_proj = nn.Linear(dim, dim * 2)  # 只使用2倍维度而非3倍
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # 缩放因子
        self.scale = head_dim ** -0.5
        
        # 正则化层
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        x: 输入张量，形状为 [batch_size, seq_len, dim]
        mask: 可选的掩码张量，形状为 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 简化的QKV生成：通过一次投影+拆分实现
        qk, v = torch.split(self.qkv_proj(x), [self.dim, self.dim], dim=-1)
        
        # 从QK中拆分出Q和K（使用不同激活函数增加区分度）
        q = F.silu(qk) * self.scale  # 融入缩放因子
        k = F.tanh(qk)
        
        # 重塑为多头结构 [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        
        # 稀疏注意力优化：只关注局部窗口
        if self.sparse:
            # 创建窗口掩码
            window_mask = torch.ones_like(attn_scores)
            for i in range(seq_len):
                # 只保留当前位置前后window_size范围内的注意力
                start = max(0, i - self.sparse_window)
                end = min(seq_len, i + self.sparse_window + 1)
                window_mask[:, :, i, :start] = 0
                window_mask[:, :, i, end:] = 0
            attn_scores = attn_scores * window_mask.to(attn_scores.device)
        
        # 应用掩码
        if mask is not None:
            # 扩展掩码维度以适应多头结构
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力到值V
        attn_output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len,self.dim)
        
        # 输出投影和残差连接
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # 残差连接 + 层归一化
        return self.norm(x + attn_output)
    
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
    
