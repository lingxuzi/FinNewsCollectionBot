import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.modules.vae_latent_mean_var import VAELambda
from ai.embedding.models import register_model

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

# --- 2. MultiModalAutoencoder (带注意力 & 强化预测 & Batch Norm) ---
@register_model(name='lstm-ae')
class MultiModalAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        ts_input_dim = config['ts_input_dim']
        ctx_input_dim = config['ctx_input_dim']
        ts_embedding_dim = config['ts_embedding_dim']
        ctx_embedding_dim = config['ctx_embedding_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers'] 
        predict_dim = config['predict_dim']
        noise_level = config['noise_level']
        noise_prob = config['noise_prob']
        dropout_rate = config['dropout']

        # 参数校验
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate 必须在 0 到 1 之间。")
            
        self.noise_level = noise_level
        self.noise_prob = noise_prob
        
        self.total_embedding_dim = ts_embedding_dim + ctx_embedding_dim
        self.use_fused_embedding = config['fused_embedding']
        self.encoder_mode = False

        # --- 分支1: 时序编码器 (LSTM) ---
        self.ts_encoder = nn.LSTM(ts_input_dim, hidden_dim, num_layers, batch_first=True)
        # self.ts_encoder_att = Attention(hidden_dim)
        self.ts_encoder_fc = VAELambda(hidden_dim, ts_embedding_dim) #nn.Linear(hidden_dim, ts_embedding_dim)
        self.ts_encoder_bn = nn.BatchNorm1d(ts_embedding_dim)  # 添加 BN

        # --- 分支2: 上下文编码器 (MLP) ---
        # 增加 Batch Normalization
        self.ctx_encoder = ResidualMLPBlock(ctx_input_dim, hidden_dim, ctx_embedding_dim, dropout_rate=0, use_batchnorm=True)

        # --- 解码器 ---
        # 时序解码器
        self.ts_decoder_fc = nn.Linear(ts_embedding_dim if not self.use_fused_embedding else self.total_embedding_dim, hidden_dim)
        self.ts_decoder_fc_bn = nn.BatchNorm1d(hidden_dim)  # 添加 BN
        self.ts_decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                  batch_first=True, dropout=dropout_rate)
        self.ts_output_layer = ResidualMLPBlock(hidden_dim, hidden_dim, ts_input_dim, dropout_rate=dropout_rate)

        # nn.init.xavier_uniform_(self.ts_decoder_fc.weight)
        # nn.init.xavier_uniform_(self.ts_output_layer.p[-1].weight)

        self.ctx_decoder = ResidualMLPBlock(ctx_embedding_dim if not self.use_fused_embedding else self.total_embedding_dim, hidden_dim, ctx_input_dim, dropout_rate=dropout_rate)
        self.predictor = ResidualMLPBlock(self.total_embedding_dim, int(hidden_dim), predict_dim, dropout_rate=dropout_rate)

        self.embedding_norm = nn.LayerNorm(self.total_embedding_dim)
        self.fusion_block = SEFusionBlock(input_dim=self.total_embedding_dim, reduction_ratio=8)

        # 初始化预测头
        self.initialize_prediction_head(self.ts_output_layer.p[-1])
        self.initialize_prediction_head(self.ctx_decoder.p[-1])
        self.initialize_prediction_head(self.predictor.p[-1])
        self.ts_encoder.flatten_parameters()
        self.ts_decoder.flatten_parameters()

        if config.get('encoder_only', False):
            self.encoder_only(True)

    def encoder_only(self, encoder=True):
        if encoder:
            self.eval()
        self.encoder_mode = encoder

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
        if self.training and self.noise_level > 0:
            if torch.rand(1).item() < self.noise_prob:
                # 添加噪声
                # 这里假设 x_ts 是一个形状为 (batch_size, seq_len, feature_dim) 的张量
                # 如果 x_ts 是一个一维时间序列，则需要调整噪声的形状
                noise = torch.normal(0, self.noise_level, size=x_ts.size(), device=x_ts.device)
                x_ts = x_ts + noise
            # if torch.rand(1).item() < self.noise_prob:
            #     # 添加噪声
            #     noise = torch.normal(0, self.noise_level, size=x_ctx.size(), device=x_ctx.device)
            #     x_ctx = x_ctx + noise
                
        # --- 编码过程 ---
        # 1. 时序编码
        ts_encoder_outputs, (ts_h_n, ts_c_n) = self.ts_encoder(x_ts)
        ts_last_hidden_state = ts_h_n[-1, :, :]
        # ts_last_hidden_state, _ = self.ts_encoder_att(ts_encoder_outputs)
        ts_embedding, ts_mean, ts_logvar = self.ts_encoder_fc(ts_last_hidden_state) 
        ts_embedding = self.ts_encoder_bn(ts_embedding)
        
        # 2. 上下文编码
        ctx_embedding = self.ctx_encoder(x_ctx)

        # # 3. 融合Embedding
        final_embedding = torch.cat([ts_embedding, ctx_embedding], dim=1)
        if not self.encoder_mode:
            if not self.use_fused_embedding:
                ts_decoder_input = self.ts_decoder_fc(ts_embedding)
            else:
                ts_decoder_input = self.ts_decoder_fc(final_embedding)
            ts_decoder_input = self.ts_decoder_fc_bn(ts_decoder_input).unsqueeze(1).repeat(1, x_ts.size(1), 1) # 应用 BN
            ts_reconstructed, _ = self.ts_decoder(ts_decoder_input)
            ts_output = self.ts_output_layer(ts_reconstructed)
        
            # 2. 上下文重构
            if not self.use_fused_embedding:
                ctx_output = self.ctx_decoder(ctx_embedding)
            else:
                ctx_output = self.ctx_decoder(final_embedding)

        # 3. Fused Embedding
        norm_embedding = self.embedding_norm(final_embedding.detach())
        norm_embedding = self.fusion_block(norm_embedding)

        # --- 3. 预测分支 ---
        predict_output = self.predictor(norm_embedding)

        if not self.encoder_mode:
            if self.training:
                return ts_output, ctx_output, predict_output, final_embedding, ts_mean, ts_logvar
            return ts_output, ctx_output, predict_output, final_embedding
        else:
            return predict_output, final_embedding

    def get_embedding(self, x_ts, x_ctx):
        """用于推理的函数，只返回融合后的embedding。"""
        with torch.no_grad():
            ts_encoder_outputs, (ts_h_n, ts_c_n) = self.ts_encoder(x_ts)
            ts_last_hidden_state = ts_h_n[-1, :, :]
            ts_embedding = self.ts_encoder_fc(ts_last_hidden_state)
            ctx_embedding = self.ctx_encoder(x_ctx)
            final_embedding = torch.cat([ts_embedding, ctx_embedding], dim=1)
        return final_embedding