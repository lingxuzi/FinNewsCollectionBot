
def read_text(path):
    with open(path, 'r') as f:
        return f.read()
    
def save_text(text, path):
    with open(path, 'w') as f:
        return f.write(text)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import torch
from torch import nn
from copy import deepcopy
class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device='cpu'):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.manual_decay = decay
        self.device = device
        self.module = self.module.to(device)
        
        self.step = 0
        self._update_decay()

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
                
    def _update_decay(self):
        self.decay = min(self.manual_decay, (1 + self.step) / (10 + self.step))
        self.step += 1

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
        self._update_decay()

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)