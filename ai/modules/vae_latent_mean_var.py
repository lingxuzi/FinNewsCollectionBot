import torch.nn as nn
import torch

class VAELambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(VAELambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Sequential(
            nn.Linear(self.hidden_size, self.latent_length),
            nn.BatchNorm1d(self.latent_length, affine=False)
        )
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        # nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        # nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        latent_mean = self.hidden_to_mean(cell_output)
        latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(latent_mean), latent_mean, latent_logvar
        else:
            return latent_mean, None, None