# model.py
import torch.nn as nn
from mamba_ssm import Mamba

# 此处代码与上一回答中的模型定义完全相同
# 为简洁起见，此处省略，请直接从上一回答复制
# LightweightMambaEncoder, LightweightMambaDecoder, LightweightMambaCodec
# ...
class LightweightMambaEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layer, latent_dim, d_state, d_conv):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.mamba_layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv) for _ in range(n_layer)]
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = x.transpose(1, 2)
        x = self.pooling(x)
        x = x.squeeze(2)
        latent_vec = self.to_latent(x)
        return latent_vec

class LightweightMambaDecoder(nn.Module):
    def __init__(self, latent_dim, d_model, n_layer, output_dim, seq_len, d_state, d_conv):
        super().__init__()
        self.seq_len = seq_len
        self.latent_to_seq = nn.Linear(latent_dim, d_model)
        self.mamba_layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv) for _ in range(n_layer)]
        )
        self.to_output = nn.Linear(d_model, output_dim)

    def forward(self, latent_vec):
        x = self.latent_to_seq(latent_vec)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        for layer in self.mamba_layers:
            x = layer(x)
        output_seq = self.to_output(x)
        return output_seq

class LightweightMambaCodec(nn.Module):
    def __init__(self, input_dim, d_model, n_layer, latent_dim, seq_len, d_state, d_conv):
        super().__init__()
        self.encoder = LightweightMambaEncoder(input_dim, d_model, n_layer, latent_dim, d_state, d_conv)
        self.decoder = LightweightMambaDecoder(latent_dim, d_model, n_layer, input_dim, seq_len, d_state, d_conv)
    
    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed_x = self.decoder(latent_vector)
        return reconstructed_x, latent_vector