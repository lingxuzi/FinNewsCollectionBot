import torch
import torch.nn as nn
import torch.nn.functional as F


class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha
    

class ELSA(nn.Module):
    def __init__(self, alpha=0.9, beta=2.0, activation=nn.ReLU()):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))
        self.activation = activation

    def forward(self, x):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = torch.sigmoid(self.beta)

        x = self.activation(x) + torch.where(x > 0, x * beta, x * alpha)
        return x
    