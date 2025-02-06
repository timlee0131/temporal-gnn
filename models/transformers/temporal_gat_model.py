import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, GraphConv, GATConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation

from models.transformers.transformer import TemporalTransformer

"""
Time and Space Model with a Transformer powered by temporal GAT (both intra and inter node attention)
"""
class TemporalGAT(nn.Module):
    def __init__(self, config, input_size: int, window: int, horizon: int, hidden_size: int, n_heads: int = 8, attention_dropout=0.6, ff_dropout=0.1, n_layers: int = 1):
        super(TemporalGAT, self).__init__()

        self.temporal_transformer = TemporalTransformer(config=config, input_size=input_size, hidden_size=hidden_size, window=window, horizon=horizon, num_heads=n_heads, num_layers=n_layers, attention_dropout=attention_dropout, feedforward_dropout=ff_dropout)
        
        self.space_nn = nn.ModuleList([
            GATConv(hidden_size, hidden_size, heads=n_heads, dropout=attention_dropout, edge_dim=0),
            GATConv(hidden_size, hidden_size, heads=1, concat=False, dropout=attention_dropout, edge_dim=0)
            ])
        
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)
    
    def forward(self, x, edge_index, edge_weight):
        h = self.temporal_transformer(x, edge_index, edge_weight)
        
        for i, conv in enumerate(self.space_nn):
            h, _ = conv(h, edge_index, edge_weight)
            if i < len(self.space_nn) - 1:
                h = F.elu(h)
        
        x_out = self.decoder(h)
        x_horizon = self.rearrange(x_out)
        
        return x_horizon