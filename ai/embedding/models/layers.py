import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.modules.activations import ELSA

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)
    def forward(self, lstm_out):
        # lstm_out 的 shape: (batch_size, sequence_length, hidden_size)
        # 计算 Attention 权重
        attention_scores = self.attention_weights(lstm_out)
        # attention_scores 的 shape: (batch_size, sequence_length, 1)
        attention_scores = torch.tanh(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights 的 shape: (batch_size, sequence_length, 1)
        # 将 Attention 权重应用于 LSTM 输出
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # context_vector 的 shape: (batch_size, hidden_size)
        return context_vector, attention_weights.squeeze(2) # 去掉最后一维，方便后续使用

# --- 1. 轻量级注意力模块 (不变) ---
class SEFusionBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for feature fusion.
    Applies lightweight self-attention to a fused feature vector.
    """
    def __init__(self, input_dim: int, reduction_ratio: int = 16):
        """
        Args:
            input_dim (int): The dimension of the fused input vector.
            reduction_ratio (int): The factor by which to reduce the dimension in the bottleneck MLP.
        """
        super().__init__()
        bottleneck_dim = max(input_dim // reduction_ratio, 4) # 保证瓶颈层不至于太小
        
        self.se_module = nn.Sequential(
            # Squeeze: 使用一个线性层将维度降低
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            # Excite: 再用一个线性层恢复到原始维度
            nn.Linear(bottleneck_dim, input_dim, bias=False),
            # Sigmoid将输出转换为0-1之间的注意力分数
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): The fused input tensor of shape [batch_size, input_dim].
        ts_output_layer
        Returns:
            Tensor: The re-weighted feature tensor of the same shape.
        """
        # x是我们的融合embedding
        
        # se_module(x) 会输出每个特征维度的注意力权重
        attention_weights = self.se_module(x)
        
        # 将原始特征与注意力权重相乘，进行重标定
        reweighted_features = x * attention_weights
        
        return reweighted_features
    

class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, act=nn.Hardswish, use_batchnorm=True, elsa=False):
        super().__init__()
        self.p = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=not use_batchnorm),
            nn.LayerNorm(hidden_dim) if use_batchnorm else nn.Identity(),
            act() if not elsa else ELSA(activation=act()),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        # 如果输入和输出维度不同，则需要一个跳跃连接的线性投影
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.p(x)
        
        # 将残差加到输出上
        return out + residual

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, act=nn.Hardswish, use_batchnorm=True, dropout_rate=0, elsa=False, residual=True):
        super().__init__()
        self.p = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=not use_batchnorm),
            nn.LayerNorm(output_dim) if use_batchnorm else nn.Identity(),
            act() if not elsa else ELSA(activation=act()),
            nn.Dropout(dropout_rate, inplace=True) if dropout_rate > 0 else nn.Identity(),
        )
        # 如果输入和输出维度不同，则需要一个跳跃连接的线性投影
        self.residual = residual
        if self.residual:
            self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        if self.residual:
            residual = self.shortcut(x)
            
            out = self.p(x)
            
            # 将残差加到输出上
            return out + residual
        else:
            return self.p(x)

class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, act=nn.ReLU, confidence=False):
        super().__init__()
        self.confidence = confidence
        self.p = ResidualMLP(input_dim, input_dim // 2, act, False, dropout_rate, elsa=True, residual=False)

        self.head_fc = nn.Linear(input_dim // 2, output_dim)
        if confidence:
            self.head_logvar = nn.Linear(input_dim // 2, output_dim)
        
        # self.init_parameters(self.p)

    def init_parameters(self, m):
        for name, module in m.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.p(x)
        x = self.head_fc(x)
        if self.confidence:
            logvar = self.head_logvar(x)
            return x, logvar
        return x
    
class NormedPredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, act=nn.SiLU, confidence=False):
        super().__init__()
        self.confidence = confidence
        self.p = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            act(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        self.head_fc = nn.Linear(input_dim, output_dim)
        if confidence:
            self.head_logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.p(x)
        x = self.head_fc(x)
        if self.confidence:
            logvar = self.head_logvar(x)
            return x, logvar
        return x

class ALSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, embedding_dim, gru=False, kl=False, dropout=0.1):
        super().__init__()
        self.gru = gru
        self.kl = kl
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) if not gru else nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # self.lstm.activation = nn.ReLU(inplace=True)
        self.embedding_fc = nn.Linear(hidden_dim * 2, embedding_dim, bias=True)
        if kl:
            self.embedding_logvar = nn.Linear(hidden_dim * 2, embedding_dim, bias=True)
        self.att_net = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim / 2)),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Tanh(),
            nn.Linear(in_features=int(hidden_dim / 2), out_features=1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        if not self.gru:
            out, (h, c) = self.lstm(x)
        else:
            out, h = self.lstm(x)
        attention_score = self.att_net(out)
        out_att = out * attention_score
        out_att = torch.sum(out_att, dim=1)
        embedding = self.embedding_fc(torch.cat([h[-1], out_att], dim=1))
        if self.kl:
            logvar = self.embedding_logvar(torch.cat([h[-1], out_att], dim=1))
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps.mul(std).add_(embedding), embedding, logvar
        return embedding, None, None

class ALSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, embedding_dim, gru=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = gru
        
        if not gru:
            # 将 embedding 映射回 LSTM 的初始隐藏状态和细胞状态
            self.embedding_to_hidden = nn.Linear(embedding_dim, num_layers * hidden_dim)
            self.embedding_to_cell = nn.Linear(embedding_dim, num_layers * hidden_dim)
        else:
            self.embedding_to_hidden = nn.Linear(embedding_dim, num_layers * hidden_dim)
        
        # 解码器LSTM
        # 它的输入维度 output_dim 应该和原始序列的 input_dim 一致
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True) if not gru else nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # self.lstm.activation = nn.ReLU(inplace=True)
        # 将LSTM的输出映射回原始数据维度
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding, seq_len):
        self.lstm.flatten_parameters()
        # embedding shape: (batch_size, embedding_dim)
        
        h_0 = self.embedding_to_hidden(embedding)
        h_0 = h_0.view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        if not self.gru:
            # 1. 从 embedding 生成初始 h_0 和 c_0
            # -> (batch_size, num_layers * hidden_dim)
            c_0 = self.embedding_to_cell(embedding)
            # -> (batch_size, num_layers, hidden_dim) -> (num_layers, batch_size, hidden_dim)
            c_0 = c_0.view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        
        # 2. 将目标序列和初始状态送入LSTM
        decoder_input = torch.zeros(embedding.size(0), seq_len, self.hidden_dim, device=embedding.device)
        if not self.gru:
            decoder_outputs, _ = self.lstm(decoder_input, (h_0, c_0))
        else:
            decoder_outputs, _ = self.lstm(decoder_input, h_0)
        # decoder_outputs shape: (batch_size, seq_len, hidden_dim)
        
        # 3. 将每个时间步的输出映射回原始维度
        reconstructed_x = self.fc_out(decoder_outputs)
        # reconstructed_x shape: (batch_size, seq_len, output_dim)
        
        return reconstructed_x