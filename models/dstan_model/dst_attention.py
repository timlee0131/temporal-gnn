import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_softmax

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

"""
Temporal Attention Network (TA) module
"""
class TA(MessagePassing):
    def __init__(self, in_channels, hidden_channels, window, num_heads=8, dropout=0.6):
        super().__init__(aggr='add') 
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.window = window
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.q_self = nn.Linear(in_channels, hidden_channels * num_heads)
        self.k_self = nn.Linear(in_channels, hidden_channels * num_heads)
        self.v_self = nn.Linear(in_channels, hidden_channels * num_heads)
        
        self.o_self = nn.Linear(hidden_channels * num_heads, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        causal_mask = torch.triu(torch.ones(window, window), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)
        
        self.scale = hidden_channels ** 0.5
    
    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=x)
    
    def message(self, x_j):
        return x_j
    
    def update(self, aggr_out, x):
        B, T, N, D = x.size()
        
        Q = self.q_self(x).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        K = self.k_self(x).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        V = self.v_self(x).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        
        e = (Q @ K.mT) / self.scale
        e = e.masked_fill(self.causal_mask, float('-inf'))
        
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        out = attention @ V
        out = out.transpose(1, 3).contiguous().view(B, T, N, self.hidden_channels * self.num_heads)
        out = self.o_self(out)
        
        return out + aggr_out

"""
Dynamic-Spatial Temporal Attention (DSTA) module
- uses similar cross-attention mechanism for dynamic spatial learning like in traverseNet, imputation model (Cini), etc.
- does NOT use hiararchical attention for inter-neigbor weighting 
"""
class MessagePassingDSTA(MessagePassing):
    def __init__(self, in_channels, hidden_channels, window, num_heads=8, dropout=0.6, cross_dropout=0.6):
        super().__init__(aggr='add') 
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.window = window
        self.num_heads = num_heads
        self.dropout = dropout
        self.cross_dropout = cross_dropout
        
        self.q_cross = nn.Linear(in_channels, hidden_channels * num_heads)
        self.k_cross = nn.Linear(in_channels, hidden_channels * num_heads)
        self.v_cross = nn.Linear(in_channels, hidden_channels * num_heads)
        
        self.q_self = nn.Linear(in_channels, hidden_channels * num_heads)
        self.k_self = nn.Linear(in_channels, hidden_channels * num_heads)
        self.v_self = nn.Linear(in_channels, hidden_channels * num_heads)
        
        self.o_self = nn.Linear(hidden_channels * num_heads, hidden_channels)
        self.o_cross = nn.Linear(hidden_channels * num_heads, hidden_channels)
        
        # simple lienar projection to combine temporal and dynamic-spatial attention
        self.o_agg = nn.Linear(hidden_channels * 2, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(cross_dropout)
        
        causal_mask = torch.triu(torch.ones(window, window), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)
        
        self.scale = hidden_channels ** 0.5
    
    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=x)
    
    def message(self, x_j, x_i, edge_weight):
        B, T, N, D = x_j.size()
        
        Q = self.q_cross(x_i).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        K = self.k_cross(x_j).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        V = self.v_cross(x_j).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        
        e = (Q @ K.mT) / self.scale
        e = e.masked_fill(self.causal_mask, float('-inf'))
        
        attention = F.softmax(e, dim=-1)
        # attention = self.dropout(attention)
        attention = self.cross_dropout(attention)
        
        out = attention @ V
        out = out.transpose(1, 3).contiguous().view(B, T, N, self.hidden_channels * self.num_heads)
        out = self.o_cross(out)
        
        return edge_weight.view(1,1,-1,1) * out
    
    def update(self, aggr_out, x):
        B, T, N, D = x.size()
        
        Q = self.q_self(x).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        K = self.k_self(x).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        V = self.v_self(x).view(B, T, N, self.num_heads, self.hidden_channels).transpose(1, 3)
        
        e = (Q @ K.mT) / self.scale
        e = e.masked_fill(self.causal_mask, float('-inf'))
        
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        out = attention @ V
        out = out.transpose(1, 3).contiguous().view(B, T, N, self.hidden_channels * self.num_heads)
        out = self.o_self(out)
        
        out = torch.cat([out, aggr_out], dim=-1)
        out = self.o_agg(out)
        
        return out

"""
Message Passing Dynamic-Spatial Temporal Attention version 2 (DSTAv2)
- uses self-loops to handle self-attention and cross-attention simulataneously
- uses built-in nn.MultiheadAttention (did not see any improvements in time or performance)
"""
class MessagePassingDSTAv2(MessagePassing):
    def __init__(self, in_channels, hidden_channels, window, num_heads=8, dropout=0.6, cross_dropout=0.6):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.window = window
        self.num_heads = num_heads

        # Linear projection to combine outputs from self- and cross-attention.
        self.out_proj = nn.Linear(hidden_channels * 2, hidden_channels)

        # Create a causal mask for the temporal window. Shape: (window, window)
        causal_mask = torch.triu(torch.ones(window, window), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)

        self.st_cross_att = nn.MultiheadAttention(embed_dim=hidden_channels,
                                                 num_heads=num_heads,
                                                 dropout=cross_dropout,
                                                 batch_first=True)
        self.t_self_att = nn.MultiheadAttention(embed_dim=hidden_channels,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)

    def forward(self, x, edge_index, edge_weight):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        return self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=x)

    def message(self, x_j, x_i, edge_weight):
        # x_j and x_i are assumed to be of shape (B, T, N, D).
        B, T, N, _ = x_j.size()
        
        # Reshape to combine B and N into the batch dimension so that temporal dimension T is the sequence.        
        q = x_i.transpose(1,2).contiguous().view(B * N, T, self.hidden_channels)
        k = x_j.transpose(1,2).contiguous().view(B * N, T, self.hidden_channels)
        v = x_j.transpose(1,2).contiguous().view(B * N, T, self.hidden_channels)
        
        # Apply PyTorch's official multi-head attention with the causal mask.
        # The mask prevents each time point from attending to future time steps.
        attn_output, _ = self.st_cross_att(query=q,
                                         key=k,
                                         value=v,
                                         attn_mask=self.causal_mask)
        # Restore original dimensions: (B, T, N, hidden_channels)
        attn_output = attn_output.view(B, N, T, self.hidden_channels).transpose(1,2)
        
        return edge_weight.view(1, 1, -1, 1) * attn_output

    # def update(self, aggr_out, x):
    #     # Self attention within each node’s temporal window.
    #     B, T, N, _ = x.size()
        
    #     # Reshape to combine B and N into the batch dimension so that temporal dimension T is the sequence.        
    #     q = x.transpose(1,2).contiguous().view(B * N, T, self.hidden_channels)
    #     k = x.transpose(1,2).contiguous().view(B * N, T, self.hidden_channels)
    #     v = x.transpose(1,2).contiguous().view(B * N, T, self.hidden_channels)
        
    #     attn_output, _ = self.t_self_att(query=q,
    #                                     key=k,
    #                                     value=v,
    #                                     attn_mask=self.causal_mask)
    #     # Restore original dimensions: (B, T, N, hidden_channels)
    #     attn_output = attn_output.view(B, N, T, self.hidden_channels).transpose(1,2)
        
    #     # Combine the self-attended features with the outputs from cross-attention.
    #     out = torch.cat([attn_output, aggr_out], dim=-1)
    #     out = self.out_proj(out)
    #     return out

class DecoderDSTA(nn.Module):
    def __init__(self, hidden_channels, out_channels, horizon, num_heads=8, dropout=0.6, ff_dropout=0.1, causal=False):
        super(DecoderDSTA, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.horizon = horizon
        
        self.dec_att = nn.MultiheadAttention(embed_dim=hidden_channels,
                                             num_heads=num_heads,
                                             dropout=dropout,
                                             batch_first=True)
        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        self.out_proj_linear = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, query=None):
        B, T, N, _ = x.size()
        
        if query is None:
            query = x
            self.horizon = T
        
        # Reshape to combine B and N into the batch dimension so that temporal dimension T is the sequence.        
        q = query.transpose(1,2).contiguous().view(B * N, self.horizon, self.hidden_channels)
        k = x.transpose(1,2).contiguous().view(B * N, T, self.hidden_channels)
        v = x.transpose(1,2).contiguous().view(B * N, T, self.hidden_channels)
        
        attn_output, _ = self.dec_att(query=q,
                                        key=k,
                                        value=v)
        # Restore original dimensions: (B, T, N, hidden_channels)
        attn_output = attn_output.view(B, N, self.horizon, self.hidden_channels).transpose(1,2)
        
        # out = self.out_proj(attn_output)
        out = self.out_proj_linear(attn_output)
        
        return out

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, window, horizon):
        super(SimpleDecoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.horizon = horizon
        
        self.fc = nn.Linear(window * hidden_channels, horizon * out_channels)

    def forward(self, x):
        # x shape: (B, T, N, F)
        B, T, N, D = x.shape

        x = x.transpose(1, 2).contiguous().view(B, N, T * D)
        x = self.fc(x)
        
        # Reshape back to (B, N, horizon, F)
        x = x.view(B, N, self.horizon, self.out_channels)
        x = x.transpose(1, 2)
        
        return x