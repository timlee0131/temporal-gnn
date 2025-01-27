import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         # Create a positional encoding matrix of shape (max_len, d_model)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
#         pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
#         pe = pe.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         print(x.size(), self.pe[:, :x.size(1)].size())
#         # x shape: (batch_size, num_nodes/sequence_length, d_model)
#         # Match the positional encoding to the input's sequence length
#         seq_len = x.size(1)  # Get the number of nodes/sequence length
#         return x + self.pe[:, :seq_len]  # Add positional encoding (broadcast over batch size)

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



# feature vectorization approach (flatten the temporal dim along the feature dim)
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, horizon, num_heads=8, dropout=0.6):
        super(SelfAttention, self).__init__()
        
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

    def forward(self, x):
        b, n , tf = x.size()
        f = tf // self.window_size

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(b, n, self.num_heads, self.hidden_size).transpose(1, 2)
        K = K.view(b, n, self.num_heads, self.hidden_size).transpose(1, 2)
        V = V.view(b, n, self.num_heads, self.hidden_size).transpose(1, 2)
        
        # e = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        e = (Q @ K.mT) / self.scale
        attention = F.softmax(e, dim=-1)
        
        # Apply dropout to attention weights
        attention = self.dropout(attention)
        
        out = attention @ V
        
        # combine the heads
        out = out.transpose(1, 2).contiguous().view(b, n, self.hidden_size * self.num_heads)
        out = self.out(out)
        
        return out
    