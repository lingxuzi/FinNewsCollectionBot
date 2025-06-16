import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 多尺度时序特征提取器 ---
class MultiScaleTemporalExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        
        # 不同时间尺度的特征提取
        self.short_term = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.medium_term = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim//2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.long_term = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim//2, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # 特征融合
        self.fusion = nn.Conv1d(hidden_dim*3, hidden_dim, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim//4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim//4, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        
        # 多尺度特征提取
        short_feat = self.short_term(x)
        medium_feat = self.medium_term(x)
        long_feat = self.long_term(x)
        
        # 特征融合
        combined = torch.cat([short_feat, medium_feat, long_feat], dim=1)
        fused = self.fusion(combined)
        
        # 通道注意力
        attn = self.attention(fused)
        enhanced = fused * attn
        
        return enhanced.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]

# --- 2. 股票时序解码器 ---
class StockTemporalDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, seq_len):
        super().__init__()
        
        self.seq_len = seq_len
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        # 分层解码结构
        self.decoder = nn.Sequential(
            # 捕获短期依赖
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            # 捕获中期依赖
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            # 最终输出层
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )
        
        # 残差连接投影
        self.residual_proj = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else None

    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size = x.shape[0]
        
        # 扩展为序列
        x_expanded = self.proj(x).unsqueeze(2).repeat(1, 1, self.seq_len)  # [batch_size, hidden_dim, seq_len]
        
        # 解码
        decoded = self.decoder(x_expanded)
        
        # 残差连接（如果需要）
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x.unsqueeze(2).repeat(1, 1, self.seq_len))
            decoded += x_residual
            
        return decoded.permute(0, 2, 1)  # [batch_size, seq_len, output_dim]

# --- 3. 轻量级注意力模块 ---
class LightweightAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        
        return out

# --- 4. 特征融合模块 ---
class FeatureFusion(nn.Module):
    def __init__(self, ts_dim, ctx_dim, out_dim):
        super().__init__()
        self.ts_proj = nn.Linear(ts_dim, out_dim)
        self.ctx_proj = nn.Linear(ctx_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(out_dim*2, out_dim),
            nn.Sigmoid()
        )

    def forward(self, ts_feat, ctx_feat):
        ts_proj = self.ts_proj(ts_feat)
        ctx_proj = self.ctx_proj(ctx_feat)
        
        gate_input = torch.cat([ts_proj, ctx_proj], dim=1)
        fusion_gate = self.gate(gate_input)
        
        fused_feat = ts_proj * fusion_gate + ctx_proj * (1 - fusion_gate)
        return fused_feat

# --- 5. 股票多模态自编码器 ---
class StockMultiModalAutoencoder(nn.Module):
    def __init__(self, ts_input_dim, ctx_input_dim, embedding_dim=64, 
                 hidden_dim=128, seq_len=30, predict_dim=1, dropout_rate=0.1):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        
        # --- 时序编码器 ---
        self.ts_encoder = MultiScaleTemporalExtractor(ts_input_dim, hidden_dim)
        
        # 时序特征池化和投影
        self.ts_pooling = nn.AdaptiveAvgPool1d(1)
        self.ts_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # 时序注意力
        self.ts_attention = LightweightAttention(hidden_dim, num_heads=4)
        
        # --- 上下文编码器 ---
        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_input_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.ctx_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # --- 特征融合 ---
        self.feature_fusion = FeatureFusion(embedding_dim, embedding_dim, embedding_dim)
        
        # --- 时序解码器 ---
        self.ts_decoder = StockTemporalDecoder(embedding_dim, ts_input_dim, hidden_dim, seq_len)
        
        # --- 上下文解码器 ---
        self.ctx_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, ctx_input_dim)
        )
        
        # --- 预测器 ---
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, predict_dim)
        )
        
        # 初始化预测头
        self.initialize_prediction_head(self.ts_decoder.decoder[-1])
        self.initialize_prediction_head(self.ctx_decoder[-1])
        self.initialize_prediction_head(self.predictor[-1])

    def initialize_prediction_head(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x_ts, x_ctx):
        # x_ts: [batch_size, seq_len, ts_input_dim]
        # x_ctx: [batch_size, ctx_input_dim]
        
        # --- 编码过程 ---
        # 1. 时序编码
        batch_size, seq_len, _ = x_ts.shape
        
        # 多尺度时序特征提取
        ts_features = self.ts_encoder(x_ts)  # [batch_size, seq_len, hidden_dim]
        
        # 应用时序注意力
        enhanced_ts_features = self.ts_attention(ts_features)
        
        # 时序特征池化
        ts_features_pooled = self.ts_pooling(enhanced_ts_features.permute(0, 2, 1)).squeeze(-1)  # [batch_size, hidden_dim]
        ts_embedding = self.ts_proj(ts_features_pooled)  # [batch_size, embedding_dim]
        
        # 2. 上下文编码
        ctx_features = self.ctx_encoder(x_ctx)
        ctx_embedding = self.ctx_proj(ctx_features)  # [batch_size, embedding_dim]
        
        # 3. 特征融合
        fused_embedding = self.feature_fusion(ts_embedding, ctx_embedding)
        
        # --- 解码过程 ---
        # 1. 时序重构
        ts_output = self.ts_decoder(fused_embedding)
        
        # 2. 上下文重构
        ctx_output = self.ctx_decoder(fused_embedding)
        
        # 3. 预测
        predict_output = self.predictor(fused_embedding)
        
        return ts_output, ctx_output, predict_output, fused_embedding

    def get_embedding(self, x_ts, x_ctx):
        """用于推理的函数，只返回融合后的embedding。"""
        with torch.no_grad():
            batch_size, seq_len, _ = x_ts.shape
            
            ts_features = self.ts_encoder(x_ts)
            enhanced_ts_features = self.ts_attention(ts_features)
            ts_features_pooled = self.ts_pooling(enhanced_ts_features.permute(0, 2, 1)).squeeze(-1)
            ts_embedding = self.ts_proj(ts_features_pooled)
            
            ctx_features = self.ctx_encoder(x_ctx)
            ctx_embedding = self.ctx_proj(ctx_features)
            
            fused_embedding = self.feature_fusion(ts_embedding, ctx_embedding)
            
        return fused_embedding