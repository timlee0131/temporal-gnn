import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
from einops.layers.torch import Rearrange  # reshape data with Einstein notation

import torch_geometric

from models.transformersV2.attention import TemporalAttention, DynamicSpatioTemporalAttention
from models.transformersV2.util import build_combined_mask

"""
Dynamic Spatio-Temporal Attention Network V1: self-attention and cross-attention combined
Dynamic Spatio-Temporal Attention Network V2: self-attention and cross-attention separated and weighted with parameter
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
    
    def forward(self, x, edge_index, edge_weight):
        mask = build_combined_mask(edge_index, edge_weight, self.num_nodes, self.window_size, mask_self=False).to(x.device)
        
        x = self.spatial_attention(x, mask)
        x = self.out(x)
        
        x = self.decoder(x)
        x = self.rearrange(x)
        
        return x

class DSTANv2(nn.Module):
    def __init__(self, config, input_size, hidden_size, num_nodes, window_size, horizon, num_heads=8, dropout=0.6):
        super(DSTANv2, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.temporal_attention = TemporalAttention(input_size, hidden_size, window_size, num_heads, dropout)
        self.spatial_attention = DynamicSpatioTemporalAttention(input_size, hidden_size, num_nodes, window_size, num_heads, dropout)
        
        self.out = nn.Linear(hidden_size * 2, hidden_size)
        
        self.decoder = nn.Linear(hidden_size, input_size)
        self.rearrange = Rearrange('b n t f -> b t n f', t=horizon)
    
    def forward(self, x):
        mask = build_combined_mask(self.edge_index, self.edge_weight, self.num_nodes, self.window_size, mask_self=True).to(x.device)
        
        x_self = self.temporal_attention(x)
        x_cross = self.spatial_attention(x, mask)
        
        x_combined = torch.cat([x_self, x_cross], dim=-1)
        x_out = self.out(x_combined)
        
        x_out = self.decoder(x_out)
        x_out = self.rearrange(x_out)
        
        return x_out