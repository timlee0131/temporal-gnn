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
- full softmax attention for intra-node
- inter-node attention similar to GAT
"""
class DSTAfullConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, window, num_heads=8, dropout=0.6, negative_slope=0.2):
        super().__init__(aggr='add') 
        
        assert hidden_channels % num_heads == 0
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.window = window
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        self.q = nn.Linear(in_channels, hidden_channels)
        self.k = nn.Linear(in_channels, hidden_channels)
        self.v = nn.Linear(in_channels, hidden_channels)
        self.o = nn.Linear(hidden_channels, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        causal_mask = torch.triu(torch.ones(window, window), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)
        
        self.scale = self.head_dim ** 0.5
        
        # inter-node attention (similar to GAT)
        self.w = nn.Linear(in_channels, hidden_channels)
        self.att = nn.Parameter(torch.Tensor(1, self.num_heads, 2 * self.head_dim * self.window))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        return self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=x, Q=Q, K=K, V=V)
    
    def message(self, x_j, x_i, Q_i, K_j, V_j, edge_weight):
        B, T, N, D = x_j.size()
        
        # intra node attention
        Q_i = Q_i.view(B, T, N, self.num_heads, self.head_dim).transpose(1, 3)
        K_j = K_j.view(B, T, N, self.num_heads, self.head_dim).transpose(1, 3)
        V_j = V_j.view(B, T, N, self.num_heads, self.head_dim).transpose(1, 3)
        
        print("Q_i, K_j, V_j: ", Q_i.size(), K_j.size(), V_j.size())
        
        e = (Q_i @ K_j.mT) / self.scale
        e = e.masked_fill(self.causal_mask, float('-inf'))
        
        attention = F.softmax(e, dim=-1)
        # attention = self.dropout(attention)
        attention = self.dropout(attention)
        
        out = attention @ V_j
        # out = out.transpose(1, 3).contiguous().view(B, T, N, self.head_dim * self.num_heads)
        # out = self.o(out)
        out = out.transpose(1,3).contiguous()
        
        # # inter-node attention  B, H, N, T, f
        node_i = Q_i.transpose(1,2).contiguous().view(B, N, self.num_heads, T * self.head_dim)
        node_j = K_j.transpose(1,2).contiguous().view(B, N, self.num_heads, T * self.head_dim)
        
        wh = torch.cat([node_i, node_j], dim=-1)
        
        e = F.leaky_relu(wh, self.negative_slope)
        e = (e * self.att).sum(dim=-1)
        
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)
        
        h_prime = (alpha.unsqueeze(1).unsqueeze(-1) * out).view(B, T, N, self.hidden_channels)
        return h_prime 
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)

class DSTAConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, window, num_heads=8, dropout=0.6, negative_slope=0.2):
        super(DSTAConv, self).__init__(aggr='add')
        
        assert hidden_channels % num_heads == 0
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.window = window
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # intra-node attention
        
        # inter-node attention (similar to GAT)
        self.w = nn.Linear(in_channels, hidden_channels)
        self.att = nn.Parameter(torch.Tensor(1, self.num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        x = self.w(x)
        return self.propagate(edge_index, edge_weight=edge_weight, x=x)
    
    def message(self, x_j, x_i, edge_weight):
        B, N, D = x_i.size()
        
        x_i = x_i.view(B, N, self.num_heads, self.head_dim)
        x_j = x_j.view(B, N, self.num_heads, self.head_dim)
        
        # intra-node attention
        
        intra_i = x_i.transpose(1,2).contiguous().view(B, self.num_heads, N, self.head_dim).unsqueeze(-1)
        intra_j = x_j.transpose(1,2).contiguous().view(B, self.num_heads, N, self.head_dim).unsqueeze(-1)
        
        q = F.elu(intra_i) + 1
        k = F.elu(intra_j) + 1
        v = intra_j
        
        kv = k.mT @ v
        out = q @ kv
        
        out = out.transpose(1, 2).contiguous().squeeze(-1)
        
        # inter-node attention  
        
        wh = torch.cat([x_i, x_j], dim=-1)
        
        e = F.leaky_relu(wh, self.negative_slope)
        e = (e * self.att).sum(dim=-1)
        
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)
        
        # h_prime = (alpha.unsqueeze(-1) * x_j).view(B, N, self.hidden_channels)
        h_prime = (alpha.unsqueeze(-1) * out).view(B, N, self.hidden_channels)
        return h_prime   
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)

"""
Improving from DSTAConv by keeping the window dimension intact for intra-node attention and flattening for inter-node attention (same as v1)
"""
class DSTAv2Conv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, window, num_heads=8, dropout=0.6, negative_slope=0.2):
        super(DSTAv2Conv, self).__init__(aggr='add')
        
        assert hidden_channels % num_heads == 0
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.window = window
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # intra-node attention
        
        # inter-node attention (similar to GAT)
        self.w = nn.Linear(in_channels, hidden_channels)
        self.att = nn.Parameter(torch.Tensor(1, self.num_heads, 2 * window * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.flatten = nn.Linear(window * self.head_dim, window * 2)
        
        self.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight):
        x = self.w(x)
        return self.propagate(edge_index, edge_weight=edge_weight, x=x)
    
    def message(self, x_j, x_i, edge_weight):
        B, T, N, D = x_i.size()
        
        x_i = x_i.view(B, T, N, self.num_heads, self.head_dim).transpose(1, 3).contiguous()
        x_j = x_j.view(B, T, N, self.num_heads, self.head_dim).transpose(1, 3).contiguous()
        
        # intra-node attention
        
        q = F.elu(x_i) + 1
        k = F.elu(x_j) + 1
        v = x_j
        
        kv = k.mT @ v
        out = q @ kv
        out = out.transpose(1, 3)
        
        # inter-node attention  
        
        node_i = x_i.view(B, self.num_heads, N, T * self.head_dim).transpose(1, 2).contiguous()
        node_j = x_j.view(B, self.num_heads, N, T * self.head_dim).transpose(1, 2).contiguous()
        
        node_i = self.flatten(node_i)
        node_j = self.flatten(node_j)
        
        wh = torch.cat([node_i, node_j], dim=-1)
        
        e = F.leaky_relu(wh, self.negative_slope)
        e = (e * self.att).sum(dim=-1)
        
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)
        
        h_prime = (alpha.unsqueeze(1).unsqueeze(-1) * out).view(B, T, N, self.hidden_channels)
        return edge_weight.view(1,1,-1,1) * h_prime   
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)

class DSTAexpConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, window, num_heads=8, dropout=0.6, negative_slope=0.2, is_flat=True):
        super(DSTAexpConv, self).__init__(aggr='add')
        
        assert hidden_channels % num_heads == 0
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.window = window
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.flat = is_flat
        
        # intra-node attention
        
        # inter-node attention (similar to GAT)
        self.w = nn.Linear(in_channels, hidden_channels)
        self.att = nn.Parameter(torch.Tensor(1, self.num_heads, 2 * self.head_dim)) if self.flat else nn.Parameter(torch.Tensor(1, self.num_heads, 2 * self.head_dim * self.window))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        x = self.w(x)
        return self.propagate(edge_index, edge_weight=edge_weight, x=x)
    
    def message(self, x_j, x_i, edge_weight):
        inter_i, inter_j = None, None
        if self.flat:
            B, N, D = x_i.size()
            x_i = x_i.view(B, N, self.num_heads, self.head_dim)
            x_j = x_j.view(B, N, self.num_heads, self.head_dim)
        
        # intra-node attention
        
            intra_i = x_i.transpose(1,2).contiguous().view(B, self.num_heads, N, self.head_dim).unsqueeze(-1)
            intra_j = x_j.transpose(1,2).contiguous().view(B, self.num_heads, N, self.head_dim).unsqueeze(-1)
            
            inter_i = x_i
            inter_j = x_j
        else:
            B, T, N, D = x_i.size()
            x_i = x_i.view(B, T, N, self.num_heads, self.head_dim).transpose(1, 3).contiguous()
            x_j = x_j.view(B, T, N, self.num_heads, self.head_dim).transpose(1, 3).contiguous()
            intra_i = x_i
            intra_j = x_j
            
            inter_i = x_i.view(B, self.num_heads, N, T * self.head_dim).transpose(1, 2).contiguous()
            inter_j = x_j.view(B, self.num_heads, N, T * self.head_dim).transpose(1, 2).contiguous()
        
        q = F.elu(intra_i) + 1
        k = F.elu(intra_j) + 1
        v = intra_j
        
        kv = k.mT @ v
        out = q @ kv
        
        if self.flat:
            out = out.transpose(1, 2).contiguous().squeeze(-1)
        else:
            out = out.transpose(1, 3)
        
        # inter-node attention  
        
        wh = torch.cat([inter_i, inter_j], dim=-1)
        
        e = F.leaky_relu(wh, self.negative_slope)
        e = (e * self.att).sum(dim=-1)
        
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)
        
        # out = inter_j # comment out when using intra-node attention
        if self.flat:
            h_prime = (alpha.unsqueeze(-1) * out).view(B, N, self.hidden_channels)
            # return edge_weight.view(1,-1,1) * h_prime
            return h_prime
        else:
            h_prime = (alpha.unsqueeze(1).unsqueeze(-1) * out).view(B, T, N, self.hidden_channels)
            return h_prime 
        
        # return edge_weight.view(1,-1,1) * out.view(B, N, self.hidden_channels)
        # return edge_weight.view(1,1,-1,1) * out.view(B, T, N, self.hidden_channels)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)

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

class SimpleDecoderRearrange(nn.Module):
    def __init__(self, hidden_channels, out_channels, window, horizon):
        super(SimpleDecoderRearrange, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.horizon = horizon
        
        self.fc = nn.Linear(hidden_channels, horizon * out_channels)

    def forward(self, x):
        B, N, TF = x.size()

        x = self.fc(x)
        
        # Reshape back to (B, N, horizon, F)
        x = x.view(B, N, self.horizon, self.out_channels)
        x = x.transpose(1, 2)
        
        return x