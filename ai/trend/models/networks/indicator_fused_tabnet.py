from ai.trend.modules.factor_interaction import LightFactorFusion
from ai.trend.modules.dropout import Spatial_Dropout
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import (
    create_explain_matrix,
    create_group_matrix,
)
import torch.nn as nn
import torch
import torch.nn.functional as F

class FactorInteractTabNet(TabNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.factor_interaction = LightFactorFusion(self.embedder.post_embed_dim,)
        self.spatial_dropout = nn.Dropout1d(p=0.1)

    def forward(self, x):
        x = self.embedder(x)
        if self.training:
            x = self.spatial_dropout(x)
        return self.tabnet.forward(x)

    def forward_masks(self, x):
        x = self.embedder(x)
        if self.training:
            x = self.spatial_dropout(x)
        return self.tabnet.forward_masks(x)

class FactorInteractTabNetClassifier(TabNetClassifier):
    def _set_network(self):
        """Setup the network and explain matrix."""
        torch.manual_seed(self.seed)

        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        self.network = FactorInteractTabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.group_matrix.to(self.device),
        ).to(self.device)

        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )