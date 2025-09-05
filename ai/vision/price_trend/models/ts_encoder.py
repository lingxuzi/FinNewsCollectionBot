from ai.embedding.models.layers import ALSTMEncoder, LayerNormedResidualMLP, ResidualMLPBlock, nn, torch, F

class TSEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ts_model = ALSTMEncoder(config['ts_input_dim'], config['hidden_dim'], num_layers=config['num_layers'], embedding_dim=config['ts_embedding_dim'], gru=True, kl=False, dropout=0.1)
        self.ctx_model = ResidualMLPBlock(config['ctx_input_dim'], config['hidden_dim'], config['ctx_embedding_dim'],dropout_rate=0, use_batchnorm=True)
        # self.embedding_projector = nn.Linear(config['ts_embedding_dim'] + config['ctx_embedding_dim'], config['hidden_dim'])
        # self.embedding_norm = nn.LayerNorm(config['hidden_dim'])
        self.fusion_block = LayerNormedResidualMLP(config['ts_embedding_dim'] + config['ctx_embedding_dim'], config['embedding_dim'], dropout_rate=config['dropout'])
    
    def forward(self, x):
        ts_seq, ctx_seq = x
        ts_emb, _, _ = self.ts_model(ts_seq)
        ctx_emb = self.ctx_model(ctx_seq)
        emb = torch.cat([ts_emb, ctx_emb], dim=1)
        emb = self.fusion_block(emb)
        return emb
