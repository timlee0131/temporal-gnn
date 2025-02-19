import torch
import torch.nn as nn
import torch_geometric
from einops.layers.torch import Rearrange  # reshape data with Einstein notation

from models.dstan_model.attention import TemporalAttention, DynamicSpatioTemporalAttention
from models.dstan_model.dst_attention import TA, MessagePassingDSTA, MessagePassingDSTAv2, DecoderDSTA, SimpleDecoder
from models.dstan_model.util import build_combined_mask, build_sparse_combined_mask, rope, pe_mem_future

"""
Dynamic Spatio-Temporal Attention Network V1: self-attention and cross-attention combined (-)
Dynamic Spatio-Temporal Attention Network V2: self-attention and cross-attention separated and weighted with parameter (-)

Dynamic Spatio-Temporal Attention Network V3: Message Passing DSTA (+)
Graph Transformer DSTA: Graph Transformer power by Message Passing DSTA (+)
"""

class DSTANv1(nn.Module):
    def __init__(self, config, input_size, hidden_size, num_nodes, window_size, horizon, num_heads=8, dropout=0.6):
        super(DSTANv1, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.spatial_attention = DynamicSpatioTemporalAttention(input_size, hidden_size, num_nodes, window_size, num_heads, dropout)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.decoder = nn.Linear(hidden_size, input_size)
        self.rearrange = Rearrange('b n t f -> b t n f', t=horizon)
        self.mask = None
    
    def forward(self, x, edge_index, edge_weight):
        if self.mask is None:
            self.mask = build_combined_mask(edge_index, edge_weight, self.num_nodes, self.window_size, mask_self=True).to(x.device)
        x = self.spatial_attention(x, self.mask)
        x = self.out(x)
        
        x = self.decoder(x)
        x = self.rearrange(x)
        
        return x

"""
Temporal Attention Network (TAN)
    - by default, no neighborhood info used, only self-attention
"""
class TAN(nn.Module):
    def __init__(self, config, input_size, hidden_size, num_nodes, window_size, horizon, num_heads=8, dropout=0.6):
        super(TAN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.temporal_attention = TA(input_size, hidden_size, window_size, num_heads, dropout)
        
        self.decoder = nn.Linear(hidden_size * window_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)
        
        self.decoder_compress = nn.Linear(hidden_size, input_size)
        self.learn_horizon = nn.Linear(window_size, horizon)
    
    def forward(self, x, edge_index, edge_weight):
        x = self.temporal_attention(x, edge_index, edge_weight)
        
        # x = x.transpose(1,2).contiguous()
        # x = x.view(x.size(0), x.size(1), -1)
        
        # x_out = self.decoder(x)
        # x_out = self.rearrange(x_out)
        
        x_out = self.decoder_compress(x)
        x_out = x_out.permute(0,2,3,1)
        x_out = self.learn_horizon(x_out)
        x_out = x_out.permute(0,3,1,2)
        
        return x_out

"""
Message Passing DSTA
"""
class MP_DSTAN(nn.Module):
    def __init__(self, config, input_size, hidden_size, window_size, horizon, num_heads=8, dropout=0.6):
        super(MP_DSTAN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.mp_dsta = MessagePassingDSTA(hidden_size, hidden_size, window_size, num_heads, dropout)
        # self.mp_dsta = MessagePassingDSTAv2(hidden_size, hidden_size, window_size, num_heads, dropout)
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size * window_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)
        
        self.decoder_compress = nn.Linear(hidden_size, input_size)
        self.learn_horizon = nn.Linear(window_size, horizon)
    
    def forward(self, x, edge_index, edge_weight):
        x = self.encoder(x)
        x = rope(x)
        
        x = self.mp_dsta(x, edge_index, edge_weight)
        
        # x = x.transpose(1,2).contiguous()
        # x = x.view(x.size(0), x.size(1), -1)
        
        # x_out = self.decoder(x)
        # x_out = self.rearrange(x_out)
        
        x_out = self.decoder_compress(x)
        x_out = x_out.permute(0,2,3,1)
        x_out = self.learn_horizon(x_out)
        x_out = x_out.permute(0,3,1,2)
        
        return x_out

class MP_DSTANv2(nn.Module):
    def __init__(self, config, input_size, hidden_size, window_size, horizon, num_heads=8, dropout=0.6):
        super(MP_DSTANv2, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.horizon = horizon
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.encoder = nn.Linear(input_size, hidden_size)

        self.mp_dsta = MessagePassingDSTAv2(hidden_size, hidden_size, window_size, num_heads, dropout, dropout)
        # self.predictor = DecoderDSTA(hidden_size, input_size, horizon=horizon, num_heads=num_heads, dropout=dropout)
        self.predictor = SimpleDecoder(hidden_size, input_size, window_size, horizon)
    
    def forward(self, x, edge_index, edge_weight):
        B, T, N, _ = x.size()
        
        future_query = nn.Parameter(torch.randn(self.horizon, 1, device=x.device))
        future_query = future_query.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)
        
        x = self.encoder(x)
        future_query = self.encoder(future_query)
        
        x_enc, future_query_enc = pe_mem_future(x, future_query)
        
        memory = self.mp_dsta(x_enc, edge_index, edge_weight)
        # pred = self.predictor(memory, future_query_enc)
        pred = self.predictor(memory) # simple decoder
        
        return pred
    
class GraphTransformerDSTA(nn.Module):
    def __init__(self, config, input_size, hidden_size, window_size, horizon, num_heads=8, attention_dropout=0.6, cross_dropout=0.6, ff_dropout=0.1):
        super(GraphTransformerDSTA, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ff_dropout = ff_dropout
        
        self.transformer_rearrange = Rearrange('b t n f -> b n (t f)', t=window_size)
        self.transformer_rearrange_back = Rearrange('b n (t f) -> b t n f', t=horizon)
        
        self.transformer = nn.ModuleList([
            torch_geometric.nn.Sequential('x, edge_index, edge_weight', [
                (MessagePassingDSTA(hidden_size, hidden_size, window_size, num_heads, attention_dropout, cross_dropout), 'x, edge_index, edge_weight -> x'),
                self.transformer_rearrange,
                nn.LayerNorm(hidden_size * window_size),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_size * window_size, hidden_size * window_size),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_size * window_size, hidden_size * window_size),
                nn.LayerNorm(hidden_size * window_size),
                self.transformer_rearrange_back
            ])
            for _ in range(config.num_layers)
        ])
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size * window_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)
    
    def forward(self, x, edge_index, edge_weight):
        x = self.encoder(x)
        x = rope(x)
        
        for block in self.transformer:
            x = block(x, edge_index, edge_weight)
        
        x = x.transpose(1,2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        
        x_out = self.decoder(x)
        x_out = self.rearrange(x_out)
        
        return x_out    
