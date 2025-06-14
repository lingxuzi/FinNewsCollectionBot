import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 轻量级注意力模块 (不变) ---
class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, attention_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, attention_dim)
        self.key_proj = nn.Linear(key_dim, attention_dim)
        self.value_proj = nn.Linear(value_dim, value_dim)
        self.score_proj = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, query, keys, values):
        projected_query = self.query_proj(query).unsqueeze(1)
        projected_keys = self.key_proj(keys)
        scores = self.score_proj(torch.tanh(projected_query + projected_keys))
        attention_weights = F.softmax(scores, dim=1).squeeze(2)
        context_vector = torch.sum(attention_weights.unsqueeze(2) * values, dim=1)
        return context_vector, attention_weights
    

class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, use_batchnorm=True):
        super().__init__()
        self.p = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
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
class MultiModalAutoencoder(nn.Module):
    def __init__(self, ts_input_dim, ctx_input_dim, ts_embedding_dim, ctx_embedding_dim, 
                 hidden_dim, num_layers, predict_dim, attention_dim=64, 
                 dropout_rate=0.1):
        super().__init__()

        # 参数校验
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate 必须在 0 到 1 之间。")
            
        self.dropout_rate = dropout_rate
        
        self.total_embedding_dim = ts_embedding_dim + ctx_embedding_dim

        # --- 分支1: 时序编码器 (LSTM) ---
        self.ts_encoder = nn.LSTM(ts_input_dim, hidden_dim, num_layers, batch_first=True)    
        self.ts_encoder_fc = nn.Linear(hidden_dim, ts_embedding_dim)
        self.ts_encoder_dropout = nn.Dropout(self.dropout_rate)

        # --- 分支2: 上下文编码器 (MLP) ---
        # 增加 Batch Normalization
        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ctx_embedding_dim)
        )
        self.ctx_encoder = ResidualMLPBlock(ctx_input_dim, hidden_dim, ctx_embedding_dim, dropout_rate=0, use_batchnorm=True)

        # --- 注意力机制 ---
        self.attention = Attention(
            query_dim=self.total_embedding_dim, 
            key_dim=hidden_dim, 
            value_dim=hidden_dim, 
            attention_dim=attention_dim
        )

        # --- 解码器 ---
        # 时序解码器
        self.ts_decoder_projection = nn.Linear(self.total_embedding_dim, hidden_dim)
        
        self.ts_decoder_input_dim_with_attention = hidden_dim + hidden_dim 
        self.ts_decoder = nn.LSTM(self.ts_decoder_input_dim_with_attention, hidden_dim, num_layers, 
                                  batch_first=True)
        self.ts_output_layer = nn.Linear(hidden_dim, ts_input_dim)
        
        # 上下文解码器
        # 增加 Batch Normalization
        # self.ctx_decoder = nn.Sequential(
        #     nn.Linear(self.total_embedding_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, ctx_input_dim)
        # )
        self.ctx_decoder = ResidualMLPBlock(self.total_embedding_dim, hidden_dim, ctx_input_dim, dropout_rate=0)

        # self.predictor = nn.Sequential(
        #     nn.Linear(self.predictor_input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, predict_dim)
        # )
        self.predictor = ResidualMLPBlock(self.total_embedding_dim, int(hidden_dim), predict_dim, dropout_rate=0.)

        # 初始化预测头
        self.initialize_prediction_head(self.ts_output_layer)
        self.initialize_prediction_head(self.ctx_decoder.p[-1])
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
        # --- 编码过程 ---
        # 1. 时序编码
        ts_encoder_outputs, (ts_h_n, ts_c_n) = self.ts_encoder(x_ts)
        ts_last_hidden_state = ts_h_n[-1, :, :]
        
        ts_embedding = self.ts_encoder_fc(ts_last_hidden_state) 
        ts_embedding = self.ts_encoder_dropout(ts_embedding)
        
        # 2. 上下文编码
        ctx_embedding = self.ctx_encoder(x_ctx)

        # # 3. 融合Embedding
        final_embedding = torch.cat([ts_embedding, ctx_embedding], dim=1)

        # --- 解码过程 ---
        # 1. 时序重构
        context_vector, _ = self.attention(final_embedding, ts_encoder_outputs, ts_encoder_outputs)
        
        ts_decoder_base_input = self.ts_decoder_projection(final_embedding) 
        
        ts_decoder_input_concat = torch.cat([ts_decoder_base_input, context_vector], dim=1)
        ts_decoder_input_repeated = ts_decoder_input_concat.unsqueeze(1).repeat(1, x_ts.size(1), 1)
        
        ts_reconstructed, _ = self.ts_decoder(ts_decoder_input_repeated)
        ts_output = self.ts_output_layer(ts_reconstructed)
        
        # 2. 上下文重构
        ctx_output = self.ctx_decoder(final_embedding)

        # --- 3. 预测分支 ---
        predict_output = self.predictor(final_embedding.detach())
        
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