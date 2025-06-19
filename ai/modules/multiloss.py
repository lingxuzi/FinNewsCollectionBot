import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    """
    def __init__(self, num=2, loss_weighting=None, uncertain_loss=True):
        super(AutomaticWeightedLoss, self).__init__()
        if loss_weighting is not None:
            params = torch.tensor(loss_weighting, requires_grad=True)
        else:
            params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.uncertain_loss = uncertain_loss

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            if self.uncertain_loss:
                precision = torch.exp(-self.params[i])
                loss_sum += precision * loss + self.params[i]
            else:
                loss_sum += self.params[i] * loss
        return loss_sum


class GradNormAdjustment(nn.Module):
    def __init__(self, model, optimizer, num=3, alpha=0.16, initial_weights=None):
        super().__init__()
        self.init_losses = None
        self.model = model
        self.optimizer = optimizer
        self.l1loss = nn.L1Loss()
        self.alpha = alpha
        if initial_weights is None:
            params = torch.ones(num, requires_grad=True)
        else:
            params = initial_weights.float()
        self.params = torch.nn.Parameter(params)


    def adjust_loss_weight(self, x):
        if self.init_losses is None:
            self.init_losses = x
        for i, loss in enumerate(x):
            x[i] = loss * self.params[i]

    def __compute_grad_l2_norm(self, layers, loss):
        G = torch.autograd.grad(loss, layers, retain_graph=True, create_graph=True)
        G_norm = torch.cat([torch.norm(g, 2).unsqueeze(0) for g in G]).sum()
        return G_norm

    def norm_weights(self, x):
        num_losses = len(self.init_losses)
        # [x[n].data[:].clamp_(min=0.0) for n, _ in enumerate(x)]
        coef = num_losses / sum(max(0, l.item()) for n, l in enumerate(x))
        for i, loss in enumerate(x):
            x[i] = loss * coef

    def adjust_grad(self, x, total_loss):
        total_loss.backward(retain_graph=True)

        shared_layer = self.model._shared_layer()

        norms = []
        ratios = []
        for i, loss in enumerate(x):
            norm = self.__compute_grad_l2_norm(shared_layer, loss)
            ratio = loss / self.init_losses[i]
            norms.append(norm)
            ratios.append(ratio)
        
        avg_norm = torch.mean(torch.stack(norms))
        avg_ratio = torch.mean(torch.stack(ratios))

        invs = [r / avg_ratio for r in ratios]
        grads = [(avg_norm * inv ** self.alpha).detach() for inv in invs]

        self.optimizer.zero_grad()

        lgrad = torch.sum(torch.stack([self.l1loss(norm, grad) for norm, grad in zip(norms, grads)]))
        lgrad.backward(retain_graph=True)
        self.optimizer.step()

        self.norm_weights(x)
        return lgrad.item()

