import torch
import torch.nn as nn
import torch.nn.functional as F

class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0, reduction='mean'):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon  # 控制多项式项的影响程度
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes)
        # targets: (batch_size) 包含类别索引
        
        # 计算softmax概率
        probs = F.softmax(logits, dim=1)
        
        # 获取目标类别的概率
        target_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze()
        
        # 计算标准交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 计算PolyLoss
        poly_loss = ce_loss + self.epsilon * (1 - target_probs)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        return poly_loss