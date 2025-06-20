import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.embedding.models import register_model

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
            nn.ReLU(inplace=True),
            # Excite: 再用一个线性层恢复到原始维度
            nn.Linear(bottleneck_dim, input_dim),
            # Sigmoid将输出转换为0-1之间的注意力分数
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): The fused input tensor of shape [batch_size, input_dim].
        
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
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, act=nn.ReLU, use_batchnorm=True):
        super().__init__()
        self.p = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batchnorm else nn.Identity(),
            act(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        # 如果输入和输出维度不同，则需要一个跳跃连接的线性投影
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.p(x)
        
        # 将残差加到输出上
        return out + residual

class LSTMTransformerEncoder(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, num_lstm_layers, transformer_hidden_dim, num_heads, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_lstm_layers, batch_first=True, dropout=dropout_rate)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden_dim, nhead=num_heads, dim_feedforward=transformer_hidden_dim, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)  # 可以调整层数
        self.lstm_hidden_dim = lstm_hidden_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.lstm_to_transformer = nn.Linear(lstm_hidden_dim, transformer_hidden_dim) # 线性映射层
    def forward(self, x):
        # LSTM 部分
        lstm_output, (h_n, c_n) = self.lstm(x)
        # 将 LSTM 的输出传递到 Transformer
        # 使用线性层将 LSTM 的隐藏状态映射到 Transformer 的维度
        transformer_input = self.lstm_to_transformer(lstm_output)
        # Transformer 部分
        transformer_output = self.transformer_encoder(transformer_input)
        # 返回 Transformer 的输出和 LSTM 的隐藏状态
        return transformer_output, (h_n, c_n)
    
# --- 3. 卷积池化层 ---
class ConvolutionalFinalLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_filters, kernel_sizes, pool_size, dropout_rate):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size) for kernel_size in kernel_sizes
        ])
        self.pool_layers = nn.ModuleList([
            nn.MaxPool1d(pool_size) for _ in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters, embedding_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # Conv1d 需要 (batch_size, input_dim, seq_len)
        conv_outputs = []
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            conv_out = F.relu(conv(x))
            pooled_out = pool(conv_out)
            conv_outputs.append(pooled_out)
        # 将所有卷积池化后的结果拼接起来
        concatenated = torch.cat(conv_outputs, dim=-1)
        concatenated = F.adaptive_max_pool1d(concatenated, 1).squeeze(-1)
        # 应用 Dropout
        concatenated = self.dropout(concatenated)
        # 使用全连接层进行映射
        embedding = self.fc(concatenated)
        return embedding
    
# --- 2. MultiModalAutoencoder (带注意力 & 强化预测 & Batch Norm) ---
@register_model(name='transformer-ae')
class MultiModalAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 从配置中加载参数
        self.ts_input_dim = config['ts_input_dim']
        self.ctx_input_dim = config['ctx_input_dim']
        self.ts_embedding_dim = config['ts_embedding_dim']
        self.ctx_embedding_dim = config['ctx_embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.predict_dim = config['predict_dim']
        self.num_heads = config['num_heads']
        self.dropout_rate = config['dropout']
        self.noise_level = config['noise_level']
        self.noise_prob = config['noise_prob']
        self.total_embedding_dim = self.ts_embedding_dim + self.ctx_embedding_dim
        # 定义模型组件
        self.ts_encoder = LSTMTransformerEncoder(self.ts_input_dim, self.hidden_dim, self.num_layers, self.hidden_dim, self.num_heads, 0.)
        self.conv_final = ConvolutionalFinalLayer(self.hidden_dim, self.ts_embedding_dim, self.hidden_dim, [3, 5], 2, self.dropout_rate)  # 卷积池化层
        self.ctx_encoder = ResidualMLPBlock(self.ctx_input_dim, self.hidden_dim, self.ctx_embedding_dim, dropout_rate=0, use_batchnorm=True)  # 上下文编码器
        self.embedding_norm = nn.LayerNorm(self.total_embedding_dim)
        self.fusion_block = SEFusionBlock(input_dim=self.total_embedding_dim, reduction_ratio=8)
        self.ts_decoder_fc = nn.Linear(self.ts_embedding_dim, self.hidden_dim)
        self.ts_decoder = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, 
                                  batch_first=True)
        self.ts_output_layer = ResidualMLPBlock(self.hidden_dim, self.hidden_dim, self.ts_input_dim, act=nn.GELU, dropout_rate=self.dropout_rate)
        self.ctx_decoder = ResidualMLPBlock(self.ctx_embedding_dim, self.hidden_dim, self.ctx_input_dim, dropout_rate=self.dropout_rate)  # 上下文解码器
        self.predictor = ResidualMLPBlock(self.total_embedding_dim, int(self.hidden_dim), self.predict_dim, dropout_rate=self.dropout_rate)

        # 初始化预测头
        self.initialize_prediction_head(self.predictor.p[-1])


    def initialize_prediction_head(self, module):
        """
        Initializes the final layer of the predictor to output zero.
        This helps the model start with a strong baseline (predicting zero return).
        """
        print("🧠 Initializing prediction head for faster convergence...")
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
            print(f"   -> Linear layer {module} has been zero-initialized.")
        else:
            print(f"   -> Module {type(module)} is not a Linear layer, skipping zero-initialization.")


    def forward(self, x_ts, x_ctx):
        _, seq_len, _ = x_ts.shape
        if self.training and self.noise_level > 0:
            if torch.rand(1).item() < self.noise_prob:
                # 添加噪声
                # 这里假设 x_ts 是一个形状为 (batch_size, seq_len, feature_dim) 的张量
                # 如果 x_ts 是一个一维时间序列，则需要调整噪声的形状
                noise = torch.normal(0, self.noise_level, size=x_ts.size(), device=x_ts.device)
                x_ts = x_ts + noise
            if torch.rand(1).item() < self.noise_prob:
                # 添加噪声
                noise = torch.normal(0, self.noise_level, size=x_ctx.size(), device=x_ctx.device)
                x_ctx = x_ctx + noise
                
        # 编码过程
        ts_encoder_output, (ts_h_n, ts_c_n) = self.ts_encoder(x_ts)
        ts_embedding = self.conv_final(ts_encoder_output)  # 使用卷积池化层得到最终的 embedding
        ctx_embedding = self.ctx_encoder(x_ctx) # (batch_size, ctx_embedding_dim)
        # 在此处扩展 final_embedding 以匹配序列长度
        #  decoder 需要序列长度的输入
        #  例如，重复 final_embedding seq_len 次
        #  这假设解码器需要一个形状为 (batch_size, seq_len, total_embedding_dim) 的输入
        #  如果解码器只需要一个 (batch_size, total_embedding_dim) 的输入，则不需要此步骤
        #扩展final_embedding，创建seq_len维度
        # 解码过程
        ts_output = self.ts_decoder_fc(ts_embedding).unsqueeze(1).repeat(1, x_ts.size(1), 1)
        ts_output, _ = self.ts_decoder(ts_output)  #  <--- 解码器输入是 final_embedding
        ts_output = self.ts_output_layer(ts_output)
        ctx_output = self.ctx_decoder(ctx_embedding)
        # 预测分支

        # 融合 Embedding
        final_embedding = torch.cat([ts_embedding, ctx_embedding], dim=1).detach()  # (batch_size, total_embedding_dim)
        final_embedding = self.embedding_norm(final_embedding)
        final_embedding = self.fusion_block(final_embedding) # (batch_size, total_embedding_dim)

        predict_output = self.predictor(final_embedding)
        return ts_output, ctx_output, predict_output, final_embedding

    def get_embedding(self, x_ts, x_ctx):
        """用于推理的函数，只返回融合后的embedding。"""
        with torch.no_grad():
            ts_encoder_outputs, (ts_h_n, ts_c_n) = self.ts_encoder(x_ts)
            ts_last_hidden_state = ts_h_n[-1, :, :]
            ts_embedding = self.ts_encoder_fc(ts_last_hidden_state)
            ctx_embedding = self.ctx_encoder(x_ctx)
            final_embedding = torch.cat([ts_embedding, ctx_embedding], dim=1)
        return final_embedding