import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.modules.vae_latent_mean_var import VAELambda
from ai.embedding.models.layers import *
from ai.embedding.models import register_model

# --- 2. MultiModalAutoencoder (带注意力 & 强化预测 & Batch Norm) ---
@register_model(name='lstm-ae')
class LSTMAutoencoder(nn.Module):
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
        self.vae = config['vae']
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.predict_dim = predict_dim
        self.dropout_rate = dropout_rate
        self.encoder_mode = False

        # --- 分支1: 时序编码器 (LSTM) ---
        self.ts_encoder = nn.LSTM(ts_input_dim, hidden_dim, num_layers, batch_first=True)
        # self.ts_encoder_att = Attention(hidden_dim)
        self.ts_encoder_fc = VAELambda(hidden_dim, ts_embedding_dim, vae=config['vae']) #nn.Linear(hidden_dim, ts_embedding_dim)

        # --- 分支2: 上下文编码器 (MLP) ---
        # 增加 Batch Normalization
        self.ctx_encoder = ResidualMLPBlock(ctx_input_dim, hidden_dim, ctx_embedding_dim, dropout_rate=0, use_batchnorm=True)

        # --- 解码器 ---
        # 时序解码器
        self.ts_decoder_fc = nn.Linear(ts_embedding_dim if not self.use_fused_embedding else self.total_embedding_dim, hidden_dim)
        self.ts_decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                  batch_first=True, dropout=dropout_rate)
        self.ts_output_layer = ResidualMLPBlock(hidden_dim, hidden_dim, ts_input_dim, dropout_rate=dropout_rate)

        self.ctx_decoder = ResidualMLPBlock(ctx_embedding_dim if not self.use_fused_embedding else self.total_embedding_dim, hidden_dim, ctx_input_dim, dropout_rate=dropout_rate)
        

        self.embedding_norm = nn.LayerNorm(self.total_embedding_dim)
        self.fusion_block = nn.Sequential(
            SEFusionBlock(input_dim=self.total_embedding_dim, reduction_ratio=8),
            ResidualMLP(self.total_embedding_dim, int(hidden_dim))
        )
        self.predictor = PredictionHead(hidden_dim, predict_dim, dropout_rate=dropout_rate) #ResidualMLPBlock(self.total_embedding_dim, int(hidden_dim), predict_dim, dropout_rate=dropout_rate)
        self.return_head = PredictionHead(hidden_dim, 1, dropout_rate=dropout_rate) #ResidualMLPBlock(self.total_embedding_dim, int(hidden_dim), predict_dim, dropout_rate=dropout_rate)
        self.trend_head = PredictionHead(hidden_dim, config['trend_classes'], dropout_rate=dropout_rate) #ResidualMLPBlock(self.total_embedding_dim, int(hidden_dim), predict_dim, dropout_rate=dropout_rate)
        # self.init_parameters()

        if config.get('encoder_only', False):
            self.encoder_only(True)

    def encoder_only(self, encoder=True):
        if encoder:
            self.eval()
        self.encoder_mode = encoder

    def init_parameters(self, heads=['ts', 'ctx', 'pred']):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        self.ts_encoder.flatten_parameters()
        self.ts_decoder.flatten_parameters()
        if self.training and self.noise_prob > 0:
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
                
        # --- 编码过程 ---
        # 1. 时序编码
        ts_encoder_outputs, (ts_h_n, ts_c_n) = self.ts_encoder(x_ts)
        ts_last_hidden_state = ts_h_n[-1, :, :]
        # ts_last_hidden_state, _ = self.ts_encoder_att(ts_encoder_outputs)
        _ts_embedding, ts_mean, ts_logvar = self.ts_encoder_fc(ts_last_hidden_state)

        # if self.training:
        #     wasserstein_distance = self.compute_wasserstein_loss(ts_embedding)
        #     gradient_penalty = self.compute_gradient_penalty(ts_embedding)

        if self.training:
            if self.vae:
                ts_embedding = ts_mean
            else:
                ts_embedding = _ts_embedding
        else:
            ts_embedding = _ts_embedding
        
        # 2. 上下文编码
        ctx_embedding = self.ctx_encoder(x_ctx)

        # # 3. 融合Embedding
        final_embedding = torch.cat([ts_embedding, ctx_embedding], dim=1)
        if not self.encoder_mode:
            if not self.use_fused_embedding:
                h_state = self.ts_decoder_fc(ts_embedding)
            else:
                h_state = self.ts_decoder_fc(final_embedding)
            h_0 = torch.stack([h_state for _ in range(self.num_layers)])
            zero_input = torch.zeros_like(h_state)
            c_0 = torch.zeros_like(h_0).to(h_0.device)
            # decoder_input = h_state.unsqueeze(1).repeat(1, x_ts.size(1), 1)
            # ts_reconstructed, _ = self.ts_decoder(decoder_input)
            ts_reconstructed, _ = self.ts_decoder(zero_input.unsqueeze(1).repeat(1, x_ts.size(1), 1), (h_0, c_0))
            ts_output = self.ts_output_layer(ts_reconstructed)
        
            # 2. 上下文重构
            if not self.use_fused_embedding:
                ctx_output = self.ctx_decoder(ctx_embedding)
            else:
                ctx_output = self.ctx_decoder(final_embedding)

        # 3. Fused Embedding
        norm_embedding = self.embedding_norm(final_embedding)
        norm_embedding = self.fusion_block(norm_embedding)

        # --- 3. 预测分支 ---
        predict_output = self.predictor(norm_embedding)
        trend_output = self.trend_head(norm_embedding)
        return_output = self.return_head(norm_embedding)

        if not self.encoder_mode:
            if self.training:
                return ts_output, ctx_output, predict_output, trend_output, return_output, final_embedding, ts_mean, ts_logvar
            return ts_output, ctx_output, predict_output, trend_output, return_output, final_embedding
        else:
            return predict_output, trend_output, return_output, final_embedding

    def get_embedding(self, x_ts, x_ctx):
        """用于推理的函数，只返回融合后的embedding。"""
        with torch.no_grad():
            ts_encoder_outputs, (ts_h_n, ts_c_n) = self.ts_encoder(x_ts)
            ts_last_hidden_state = ts_h_n[-1, :, :]
            ts_embedding = self.ts_encoder_fc(ts_last_hidden_state)
            ctx_embedding = self.ctx_encoder(x_ctx)
            final_embedding = torch.cat([ts_embedding, ctx_embedding], dim=1)
        return final_embedding