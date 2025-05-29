import torch
import torch.nn as nn
import torch.nn.init as init

class FactorInteraction(nn.Module):
    def __init__(self, input_dim, rank=4):
        super().__init__()
        # 低秩分解参数
        self.U = nn.Parameter(torch.Tensor(input_dim, rank))
        self.V = nn.Parameter(torch.Tensor(rank, input_dim))
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Hardsigmoid()
        )
        self.norm = nn.LayerNorm(input_dim)
        init.xavier_normal_(self.U)
        init.xavier_normal_(self.V)
        
    def forward(self, x):
        # 低秩交叉项: x * U * V * x^T
        cross_term = torch.matmul(x, self.U)  # [batch, rank]
        cross_term = torch.matmul(cross_term, self.V)  # [batch, input_dim]
        interaction = x * cross_term  # Hadamard积
        interaction = self.norm(interaction)
        
        # 动态门控
        gate = self.gate(x)
        return gate * interaction + (1 - gate) * x

class LightweightSTG(nn.Module):
    def __init__(self, input_dim, keep_ratio=0.3):
        super().__init__()
        self.k = max(1, int(keep_ratio * input_dim))
        self.selector = nn.Linear(input_dim, input_dim)  # 单层评分网络
        # self.expander = nn.Linear(self.k, input_dim)  # 单层扩展网络
        
    def forward(self, x):
        # 计算因子重要性得分
        feature_scores = torch.sigmoid(self.selector(x))  # [B, N]
        
        # 沿特征维度选择Top-k
        _, topk_indices = torch.topk(
            feature_scores, 
            k=self.k, 
            dim=-1,  # 在特征维度操作
            largest=True
        )
        
        # feature_scores[topk_indices] = 1
        att_scores = feature_scores.clone().zero_().scatter_(-1, topk_indices, 1)
        # 稀疏特征
        return torch.multiply(att_scores, x), feature_scores
    
class LowRankInteraction(nn.Module):
    def __init__(self, input_dim, rank=2):
        super().__init__()
        self.U = nn.Parameter(torch.randn(input_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, input_dim))
        self.scale = 1.0 / (rank ** 0.5)
        
    def forward(self, x):
        # 低秩交互项
        cross_term = torch.matmul(x, self.U)     # [B, K, R]
        cross_term = torch.matmul(cross_term, self.V)  # [B, K, K]
        return x + self.scale * (x * cross_term)  # 残差连接

class DynamicResidualFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Linear(input_dim, 1)  # 单层门控
        
    def forward(self, x_orig, x_inter):
        gate = torch.sigmoid(self.gate(x_inter))
        return gate * x_inter + (1 - gate) * x_orig
    
class LightFactorFusion(nn.Module):
    def __init__(self, input_dim=64, keep_ratio=0.5):
        super().__init__()
        rank = max(1, int(0.1 * input_dim))
        self.stg = LightweightSTG(input_dim, keep_ratio)
        self.interaction = LowRankInteraction(int(input_dim), rank)
        self.fusion = DynamicResidualFusion(int(input_dim))
        
    def forward(self, x):
        x_sparse, scores = self.stg(x)
        x_inter = self.interaction(x_sparse)
        x_fused = self.fusion(x_sparse, x_inter)
        return x_fused

if __name__ == '__main__':
    # 示例数据
    x = torch.randn(32, 10)  # 32个样本，每个样本有10个特征
    model = LightFactorFusion(input_dim=10)
    output = model(x)
    print(output.shape)  # 输出 (32, 10)