import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from einops.layers.torch import Rearrange  # reshape data with Einstein notation

from models.dstan_model.attention import TemporalAttention, DynamicSpatioTemporalAttention
from models.dstan_model.dst_attention import TA, DSTAConv, DSTAv2Conv, DSTAfullConv, DSTAexpConv
from models.dstan_model.dst_attention import DecoderDSTA, SimpleDecoder, SimpleDecoderRearrange
from models.dstan_model.util import rope, pe_mem_future, pe_flat

"""
Dynamic Spatio-Temporal Attention Network V1: self-attention and cross-attention combined (-)
Dynamic Spatio-Temporal Attention Network V2: self-attention and cross-attention separated and weighted with parameter (-)

Dynamic Spatio-Temporal Attention Network V3: Message Passing DSTA (+)
Graph Transformer DSTA: Graph Transformer power by Message Passing DSTA (+)
"""

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
Message Passing DSTAN
"""
class MP_DSTAN(nn.Module):
    def __init__(self, config, input_size, hidden_size, window_size, horizon, num_heads=8, dropout=0.1):
        super(MP_DSTAN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.horizon = horizon
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.encoder = nn.Linear(input_size, hidden_size)

        self.mp_dsta = MessagePassingDSTA(hidden_size, hidden_size, window_size, num_heads, dropout)
        self.predictor = SimpleDecoder(hidden_size, input_size, window_size, horizon)
    
    def forward(self, x, edge_index, edge_weight):
        B, T, N, _ = x.size()
        
        future_query = nn.Parameter(torch.randn(self.horizon, 1, device=x.device))
        future_query = future_query.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)
        
        x = self.encoder(x)
        future_query = self.encoder(future_query)
        
        x_enc, future_query_enc = pe_mem_future(x, future_query)
        
        memory = self.mp_dsta(x_enc, edge_index, edge_weight)
        pred = self.predictor(memory) # simple decoder
        
        return pred

"""
Dynamic Spatio-Temporal Attention Network (DSTAN)
- handles node-level attention unlike MP_DSTAN approach that only performed attention along the temporal dimension of the neighbors
"""
class DSTAN(nn.Module):
    def __init__(self, config, input_size, hidden_size, window_size, horizon, num_heads=8, dropout=0.6, negative_slope=0.2, num_layers=2):
        super(DSTAN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.horizon = horizon
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_layers = num_layers

        self.encoder = nn.Linear(input_size, hidden_size)

        self.convs = nn.ModuleList([
            torch_geometric.nn.Sequential('x, edge_index, edge_weight', [
                (DSTAfullConv(hidden_size, hidden_size, window_size, num_heads, dropout, negative_slope), 'x, edge_index, edge_weight -> x'),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            ])
            for _ in range(num_layers)
        ])
        
        # self.predictor = SimpleDecoderRearrange(hidden_size, input_size, window_size, horizon)
        self.predictor = SimpleDecoder(hidden_size, input_size, window_size, horizon)
    
    def forward(self, x, edge_index, edge_weight):
        B, T, N, D = x.size()
        
        # x = x.transpose(1,2).contiguous().view(B, N, T * D)
        # x = pe_flat(x)
        x = self.encoder(x)
        
        future_query = nn.Parameter(torch.randn(self.horizon, 1, device=x.device))
        future_query = future_query.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)
    
        future_query = self.encoder(future_query)
        x, future_query_enc = pe_mem_future(x, future_query)
        
        for i, conv in enumerate(self.convs):
            x_res = x
            x = conv(x, edge_index, edge_weight)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
            
            if x.size(-1) == x_res.size(-1):
                x = x + x_res

        pred = self.predictor(x) # simple decoder
        
        return pred


"""
DSTAN Experiments

Attention with flattened time window (DSTAConv) vs. attention with time window (DSTAv2Conv)
"""
class DSTANExperiments(nn.Module):
    def __init__(self, config, input_size, hidden_size, window_size, horizon, num_heads=8, dropout=0.6, negative_slope=0.2, num_layers=2):
        super(DSTANExperiments, self).__init__()
        
        self.flat = True
        
        self.input_size = 12 if self.flat else 1
        self.hidden_size = 96 if self.flat else 8
        
        self.window_size = 12
        self.horizon = 12
        self.num_heads = 8
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_layers = num_layers

        self.encoder = nn.Linear(self.input_size, self.hidden_size)

        self.convs = nn.ModuleList([
            torch_geometric.nn.Sequential('x, edge_index, edge_weight', [
                (DSTAexpConv(self.hidden_size, self.hidden_size, self.window_size, self.num_heads, self.dropout, self.negative_slope, self.flat), 'x, edge_index, edge_weight -> x'),
                nn.LayerNorm(self.hidden_size),
                nn.Dropout(0.1)
            ])
            for _ in range(self.num_layers)
        ])
        
        self.predictor = SimpleDecoderRearrange(self.hidden_size, 1, self.window_size, self.horizon) if self.flat else SimpleDecoder(self.hidden_size, self.input_size, self.window_size, self.horizon)
    
    def forward(self, x, edge_index, edge_weight):
        B, T, N, D = x.size()
        
        if self.flat:
            x = x.transpose(1,2).contiguous().view(B, N, T * D)
            x = pe_flat(x)
            x = self.encoder(x)
        else:
            future_query = nn.Parameter(torch.randn(self.horizon, 1, device=x.device))
            future_query = future_query.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)

            x = self.encoder(x)
            future_query = self.encoder(future_query)
            x, future_query_enc = pe_mem_future(x, future_query)
        
        for i, conv in enumerate(self.convs):
            x_res = x
            x = conv(x, edge_index, edge_weight)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
            
            if x.size(-1) == x_res.size(-1):
                x = x + x_res

        pred = self.predictor(x) # simple decoder
        
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
