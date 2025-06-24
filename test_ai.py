from ai.modules.xlstm import sLSTM
import torch

if __name__ == '__main__':
    model = sLSTM(10, 20, 2, batch_first=True)
    x = torch.randn(32, 50, 10)
    h, o = model(x)
    print(h.shape)