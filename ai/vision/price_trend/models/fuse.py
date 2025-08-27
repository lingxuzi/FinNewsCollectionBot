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

class FeatureFusedAttention(nn.Module):
    def __init__(self, fused_dim, hidden_dim):
        super().__init__()
        self.projector = nn.Linear(fused_dim * 2, hidden_dim)
        self.att_net = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim, bias=False),
            nn.Sigmoid(),
            nn.Identity() #nn.Dropout(0.5)
        )

        self.final_projector = nn.Linear(hidden_dim, fused_dim)

    def forward(self, vision_features, ts_features):
        fused_features = torch.cat([vision_features, ts_features], dim=1)
        fused_features = self.projector(fused_features)

        fused_features = nn.functional.dropout(fused_features, p=0.5, training=self.training)

        att_features = self.att_net(fused_features)
        att_features = att_features * fused_features

        fused_features = self.final_projector(att_features)#torch.cat([fused_features, att_features], dim=1))

        # fused_features = nn.functional.dropout(fused_features, p=0.3, training=self.training)

        return fused_features


def get_fusing_layer(method='default', fused_dim=1024, hidden_dim=1024, **kwargs):
    if method == 'cross':
        return CrossFused(fused_dim, hidden_dim, **kwargs)
    elif method == 'default':
        return FeatureFusedAttention(fused_dim, hidden_dim, **kwargs)
    else:
        raise ValueError(f'Unknown fusing method: {method}')
