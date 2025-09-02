import torch
import torch.nn as nn

class CrossFused(nn.Module):
    def __init__(self, fused_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.v_proj = nn.Linear(fused_dim, hidden_dim)
        self.t_proj = nn.Linear(fused_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(fused_dim)
        self.out_proj = nn.Linear(hidden_dim, fused_dim)

    def forward(self, v, t):
        v, t = self.v_proj(v), self.t_proj(t)
        # 让 vision 作为 Query，TS 作为 K/V
        out, _ = self.cross_attn(query=v, key=t, value=t)
        out = self.norm1(v + out)
        # 可再对称做一次 TS→Vision，或简单 FFN
        out = self.norm2(self.out_proj(out))
        return out
    
class CrossModalAttention(nn.Module):
    def __init__(self, fused_dim, hidden_dim):
        super().__init__()
        # 共享投影层（保持参数量）
        self.vis_projector = nn.Linear(fused_dim, hidden_dim)
        self.ts_projector = nn.Linear(fused_dim, hidden_dim)
        
        # 分离的注意力分支（参数总量与原网络相当）
        self.vision_att = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),  # 输入包含时序特征用于交互
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim, bias=False),
            nn.Sigmoid()
        )
        
        self.ts_att = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),  # 输入包含视觉特征用于交互
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim, bias=False),
            nn.Sigmoid()
        )
        
        # 最终融合投影（保持原计算量）
        self.final_projector = nn.Linear(hidden_dim * 2, fused_dim)

    def forward(self, vision_features, ts_features):
        # 1. 基础投影（共享权重，减少参数）
        v_proj = self.vis_projector(vision_features)  # 视觉特征投影
        t_proj = self.ts_projector(ts_features)      # 时序特征投影
        
        # 2. 交叉注意力计算（核心改进）
        # 视觉注意力同时参考时序特征，增强交互
        vision_att_input = torch.cat([v_proj, t_proj], dim=1)  # 跨模态输入
        vision_weights = self.vision_att(vision_att_input)
        v_att = v_proj * vision_weights
        
        # 时序注意力同时参考视觉特征，增强交互
        ts_att_input = torch.cat([t_proj, v_proj], dim=1)  # 跨模态输入（顺序交换）
        ts_weights = self.ts_att(ts_att_input)
        t_att = t_proj * ts_weights
        
        # 3. 特征融合与输出
        fused = torch.cat([v_att, t_att], dim=1)
        output = self.final_projector(fused)
        
        # 残差连接保留原始特征
        return output + (vision_features + ts_features) * 0.5

class FeatureFusedAttention(nn.Module):
    def __init__(self, fused_dim, hidden_dim):
        super().__init__()
        self.projector = nn.Linear(fused_dim * 2, hidden_dim)
        self.att_net = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim, bias=False),
            nn.Sigmoid()
        )

        self.final_projector = nn.Linear(hidden_dim * 2, fused_dim)

    def forward(self, vision_features, ts_features):
        fused_features = torch.cat([vision_features, ts_features], dim=1)
        fused_features = self.projector(fused_features)

        fused_features = nn.functional.dropout(fused_features, p=0.5, training=self.training)

        att_features = self.att_net(fused_features)
        att_features = att_features * fused_features

        fused_features = self.final_projector(torch.cat([fused_features, att_features], dim=1))

        return fused_features


def get_fusing_layer(method='default', fused_dim=1024, hidden_dim=1024, **kwargs):
    if method == 'cross':
        return CrossModalAttention(fused_dim, hidden_dim, **kwargs)
    elif method == 'default':
        return FeatureFusedAttention(fused_dim, hidden_dim, **kwargs)
    else:
        raise ValueError(f'Unknown fusing method: {method}')
