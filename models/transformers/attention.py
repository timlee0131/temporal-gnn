import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# compatibility: feature vectorization approach (flatten the temporal dim along the feature dim)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

"""
Node-wise Self Attention
This class only performs intra-node self-attention
"""
class NodeWiseSelfAttention(nn.Module):
    def __init__(self, config, input_size, hidden_size, window_size, horizon, num_heads=8, dropout=0.6):
        super(NodeWiseSelfAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_per_head = hidden_size // num_heads
        
        self.window_size = window_size
        self.horizon = horizon
        self.num_heads = num_heads
        
        self.query = nn.Linear(input_size, hidden_size * num_heads)
        self.key = nn.Linear(input_size, hidden_size * num_heads)
        self.value = nn.Linear(input_size, hidden_size * num_heads)
        self.out = nn.Linear(hidden_size * num_heads, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.hidden_per_head ** 0.5
        
        # self.causal_mask = torch.triu(torch.ones(config.batch_size, self.num_heads, config.num_nodes, self.hidden_size, self.hidden_size), diagonal=1).bool()
        causal_mask = torch.triu(torch.ones(1,1, hidden_size, hidden_size), diagonal=1)
        """
        register buffer:
        - creates the mask only once during initialization, rather than recreating it every time the forward function is called (could otherwise be very expensive for batch processing)
        - automatically moves to the same device as the model's parameters
        """
        self.register_buffer('causal_mask', causal_mask.bool())

    def forward(self, x):
        B, N, tf = x.size()

        Q = self.query(x).view(B, N, self.num_heads, self.hidden_size).transpose(1, 2).unsqueeze(-1)
        K = self.key(x).view(B, N, self.num_heads, self.hidden_size).transpose(1, 2).unsqueeze(-1)
        V = self.value(x).view(B, N, self.num_heads, self.hidden_size).transpose(1, 2).unsqueeze(-1)

        # attention = Q @ K.mT / self.scale
        # attention = attention.masked_fill(self.causal_mask, float('-inf'))
        # attention = F.softmax(attention, dim=-1)
        # attention = self.dropout(attention)
        # out = (attention @ V).squeeze()
        attention = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.6, is_causal=True, scale=self.scale).squeeze()
        
        # combine the heads
        out = attention.transpose(1, 2).contiguous().view(B, N, self.hidden_size * self.num_heads)
        out = self.out(out)
        
        return out

# LEGACY
"""
Graph-wise Self Attention
Inter-node self-attention (intra-node self attention + cross attention across all other nodes)
"""
class GraphWiseSelfAttention(nn.Module):
    def __init__(self, config, input_size, hidden_size, window_size, horizon, num_heads=8, dropout=0.6):
        super(GraphWiseSelfAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.horizon = horizon
        self.num_heads = num_heads
        
        self.query = nn.Linear(input_size, hidden_size * num_heads)
        self.key = nn.Linear(input_size, hidden_size * num_heads)
        self.value = nn.Linear(input_size, hidden_size * num_heads)
        self.out = nn.Linear(hidden_size * num_heads, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.hidden_size ** 0.5
        
    def forward(self, x, mask):
        B, N, D = x.size()
        
        Q = self.query(x).view(B, N, self.num_heads, self.hidden_size).transpose(1,2)
        K = self.key(x).view(B, N, self.num_heads, self.hidden_size).transpose(1,2)
        V = self.value(x).view(B, N, self.num_heads, self.hidden_size).transpose(1,2)
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        attention = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.6, scale=self.scale)
        
        out = attention.transpose(1, 2).contiguous().view(B, N, self.hidden_size * self.num_heads)
        out = self.out(out)
        
        return out