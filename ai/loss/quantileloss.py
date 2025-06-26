import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y, y_pred):
        residual = y_pred - y
        loss = torch.max((self.quantile - 1) * residual, self.quantile * residual)
        return torch.mean(loss)