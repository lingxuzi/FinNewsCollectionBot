import torch
import torch.nn as nn

class BucketFusion(nn.Module):
    def __init__(self, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        # 可学习的分箱边界（初始化为均匀分位）
        self.bin_edges = nn.Parameter(torch.linspace(0, 1, n_bins+1))
    
    def forward(self, x):
        # 归一化到 [0,1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6)
        # 计算分箱掩码
        bin_mask = torch.bucketize(x_norm, self.bin_edges)
        # 统计每个箱的均值
        bucket_means = []
        for i in range(self.n_bins):
            mask = (bin_mask == i)
            bucket_mean = torch.where(mask, x, 0).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
            bucket_means.append(bucket_mean)
        fused = torch.stack(bucket_means, dim=1).mean(dim=1)
        return fused.unsqueeze(-1)
    

if __name__ == '__main__':
    # 示例数据
    x = torch.randn(32, 10)  # 32个样本，每个样本有10个特征
    model = BucketFusion(n_bins=5)
    output = model(x)
    print(output.shape)  # 输出 (32, 1)