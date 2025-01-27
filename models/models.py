import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, GraphConv, GATConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation

from models.transformer import TemporalTransformer

# boiler plate time then space model from tsl documentation (time: RNN (GRU) space: GCN)
class TTS_RNN_GCN(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int, hidden_size: int = 32, rnn_layers: int = 1):
        super(TTS_RNN_GCN, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size) # free params learned individually for each node (taming local effects in STGNNs, Cini et al.)

        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)
        
        self.space_nn = GraphConv(input_size=hidden_size, output_size=hidden_size)

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index, edge_weight)  # spatial processing
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon

"""
time then space model
time: transformer
space: GAT
"""
class TTS_TRF_GAT(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, window: int, horizon: int, hidden_size: int, n_heads: int = 8, attention_dropout=0.6, ff_dropout=0.1, n_layers: int = 1):
        super(TTS_TRF_GAT, self).__init__()

        # time nn
        self.temporal_transformer = TemporalTransformer(input_size=input_size, hidden_size=hidden_size, window=window, horizon=horizon, num_heads=n_heads, num_layers=n_layers, attention_dropout=attention_dropout, feedforward_dropout=ff_dropout)
        
        # space nn
        # self.space_nn = GraphConv(input_size=hidden_size, output_size=hidden_size)
        self.space_nn = nn.ModuleList([
            GATConv(hidden_size, hidden_size, heads=n_heads, dropout=attention_dropout, edge_dim=0),
            GATConv(hidden_size, hidden_size, heads=8, dropout=attention_dropout, edge_dim=0),
            GATConv(hidden_size, hidden_size, heads=1, concat=False, dropout=attention_dropout, edge_dim=0)
            ])
        
        
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)
    
    def forward(self, x, edge_index, edge_weight):
        h = self.temporal_transformer(x)
        # h = self.space_nn(h, edge_index, edge_weight)        
        for i, conv in enumerate(self.space_nn):
            h, _ = conv(h, edge_index, edge_weight)
            if i < len(self.space_nn) - 1:
                h = F.elu(h)
        
        x_out = self.decoder(h)
        x_horizon = self.rearrange(x_out)
        
        return x_horizon
        