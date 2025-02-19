import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing

from models.dstan_model.util import PositionalEncoding, build_combined_mask

"""
Currently only using MessagePassingDSTA as this is the best implementation
"""

"""
Temporal Attention O(N * T^2 * d)
"""
class TemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, num_heads=8, dropout=0.6):
        super(TemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size)

        self.self_q = nn.Linear(hidden_size, hidden_size * num_heads)
        self.self_k = nn.Linear(hidden_size, hidden_size * num_heads)
        self.self_v = nn.Linear(hidden_size, hidden_size * num_heads)

        self.out = nn.Linear(hidden_size * num_heads, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = hidden_size ** 0.5
        causal_mask = torch.triu(torch.ones(window_size, window_size), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        B, N, T, D = x.size()
        
        x = self.encoder(x)
        x = self.pe(x)
        
        Q_self = self.self_q(x).view(B, N, T, self.num_heads, self.hidden_size).transpose(2, 3)
        K_self = self.self_k(x).view(B, N, T, self.num_heads, self.hidden_size).transpose(2, 3)
        V_self = self.self_v(x).view(B, N, T, self.num_heads, self.hidden_size).transpose(2, 3)
        
        e_self = (Q_self @ K_self.mT) / self.scale
        e_self = e_self.masked_fill(self.causal_mask, float('-inf'))
        
        attention_Self = F.softmax(e_self, dim=-1)
        attention_Self = self.dropout(attention_Self)
        out_self = attention_Self @ V_self
        
        out_self = out_self.transpose(2, 3).contiguous().view(B, N, T, self.hidden_size * self.num_heads)
        out_self = self.out(out_self)
        
        return out_self

"""
Dynamic Spatio-Temporal Attention with 2 possible variants (conditioned on self_mask) O(N^2 * T^2 * d)
    1. combine self-attention and cross-attention
    2. separate self-attention and cross-attention
"""
class DynamicSpatioTemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, window_size, num_heads=8, dropout=0.6):
        super(DynamicSpatioTemporalAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size)
        
        self.cross_q = nn.Linear(hidden_size, hidden_size * num_heads)
        self.cross_k = nn.Linear(hidden_size, hidden_size * num_heads)
        self.cross_v = nn.Linear(hidden_size, hidden_size * num_heads)

        self.out = nn.Linear(hidden_size * num_heads, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = hidden_size ** 0.5
        
        # includes self-loop for self-attention + causal mask for ST attention
        # mask = torch.triu(torch.ones(num_nodes * window_size, num_nodes * window_size), diagonal=1).bool()
        # self.register_buffer('mask', mask)
    
    def forward(self, x, mask):
        x = x.permute(0, 2, 1, 3)
        B, N, T, D = x.size()
        
        x = self.encoder(x)
        x = self.pe(x)
        
        # (B, N, T, D) -> (B, N, T, num_heads, hidden_size) -> (B, num_heads, N, T, hidden_size)
        Q = self.cross_q(x).view(B, N, T, self.num_heads, self.hidden_size).transpose(1, 3)
        K = self.cross_k(x).view(B, N, T, self.num_heads, self.hidden_size).transpose(1, 3)
        V = self.cross_v(x).view(B, N, T, self.num_heads, self.hidden_size).transpose(1, 3)
        
        # (B, num_heads, N * T, hidden_size)
        Q_flat = Q.reshape(B, self.num_heads, N * T, self.hidden_size)
        K_flat = K.reshape(B, self.num_heads, N * T, self.hidden_size)
        V_flat = V.reshape(B, self.num_heads, N * T, self.hidden_size)
        
        e = Q_flat @ K_flat.mT / self.scale # (B, num_heads, N * T, N * T)
        # e = e.masked_fill(self.mask, float('-inf'))
        e = e.masked_fill(mask, float('-inf'))
        
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        out_flat = attention @ V_flat # (B, num_heads, N * T, hidden_size)
        # out = out_flat.transpose(1, 2).reshape(B, N, T, self.num_heads, self.hidden_size)
        # out = out_flat.reshape(B, self.num_heads, N, T, self.hidden_size).transpose(1,3) # (B, N, T, num_heads, hidden_size)
        
        out = out_flat.transpose(1, 2).contiguous().view(B, N, T, self.num_heads * self.hidden_size)
        out = self.out(out)
        
        return out