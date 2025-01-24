import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, GraphConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation

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