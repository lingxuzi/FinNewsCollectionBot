import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.modules.vae_latent_mean_var import VAELambda
from ai.embedding.models.layers import *
from ai.embedding.models import register_model



# --- 2. MultiModalAutoencoder (带注意力 & 强化预测 & Batch Norm) ---
@register_model(name='alstm-ae')
class ALSTMAutoencoder(nn.Module):
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
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.predict_dim = predict_dim
        self.dropout_rate = dropout_rate
        self.encoder_mode = False

        # --- 分支1: 时序编码器 (LSTM) ---
        self.ts_encoder = ALSTMEncoder(ts_input_dim, hidden_dim, num_layers, ts_embedding_dim)

        # --- 分支2: 上下文编码器 (MLP) ---
        # 增加 Batch Normalization
        self.ctx_encoder = ResidualMLPBlock(ctx_input_dim, hidden_dim, ctx_embedding_dim, dropout_rate=0, use_batchnorm=True)

        # --- 解码器 ---
        # 时序解码器
        self.ts_decoder = ALSTMDecoder(ts_input_dim, hidden_dim, num_layers, ts_embedding_dim)

        self.ctx_decoder = ResidualMLPBlock(ctx_embedding_dim if not self.use_fused_embedding else self.total_embedding_dim, hidden_dim, ctx_input_dim, dropout_rate=dropout_rate)
        

        self.embedding_norm = nn.LayerNorm(self.total_embedding_dim)
        self.fusion_block = nn.Sequential(
            SEFusionBlock(input_dim=self.total_embedding_dim, reduction_ratio=8),
            ResidualMLP(self.total_embedding_dim, int(hidden_dim))
        )
        self.predictor = PredictionHead(hidden_dim, predict_dim, act=nn.Tanh, dropout_rate=dropout_rate) #ResidualMLPBlock(self.total_embedding_dim, int(hidden_dim), predict_dim, dropout_rate=dropout_rate)
        self.return_head = PredictionHead(hidden_dim, predict_dim, act=nn.Tanh, dropout_rate=dropout_rate) #ResidualMLPBlock(self.total_embedding_dim, int(hidden_dim), predict_dim, dropout_rate=dropout_rate)
        self.trend_head = nn.Sequential(
            PredictionHead(hidden_dim, predict_dim, act=nn.Tanh, dropout_rate=dropout_rate),
            nn.Sigmoid()
        ) #ResidualMLPBlock(self.total_embedding_dim, int(hidden_dim), predict_dim, dropout_rate=dropout_rate)
        self.init_parameters()
        self.initialize_prediction_head(self.return_head.p[-1])

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
        seq_len = x_ts.size(1)
        ts_embedding = self.ts_encoder(x_ts)
        # 2. 上下文编码
        ctx_embedding = self.ctx_encoder(x_ctx)

        # # 3. 融合Embedding
        final_embedding = torch.cat([ts_embedding, ctx_embedding], dim=1)
        if not self.encoder_mode:
            if not self.use_fused_embedding:
                ts_output = self.ts_decoder(ts_embedding, seq_len)
            else:
                ts_output = self.ts_decoder(final_embedding, seq_len)
        
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
        trend_output = self.trend_head(norm_embedding)
        return_output = self.return_head(norm_embedding)

        if not self.encoder_mode:
            if self.training:
                return ts_output, ctx_output, predict_output, trend_output, return_output, final_embedding, None, None
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