import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

import torch_geometric

from models.transformers.attention import PositionalEncoding, NodeWiseSelfAttention, GraphWiseSelfAttention

class TemporalTransformer(nn.Module):
    def __init__(self, config, input_size, hidden_size, window, horizon, num_heads=8, num_layers=1, attention_dropout=0.6, feedforward_dropout=0.1):
        super(TemporalTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window = window
        self.horizon = horizon
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        
        self.encoder = nn.Linear(self.input_size * window, self.hidden_size)
        self.positional_encoding = PositionalEncoding(self.input_size * window)
        
        self.transformers = nn.ModuleList([
            torch.nn.Sequential(
                NodeWiseSelfAttention(config, self.hidden_size, self.hidden_size, window, horizon, num_heads, attention_dropout),
                nn.LayerNorm(self.hidden_size),
                nn.Dropout(feedforward_dropout),
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(feedforward_dropout),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, x, edge_index, edge_weight):
        b, t, n, f = x.size()
        x = x.permute(0,2,1,3).reshape(b, n, t*f)
        
        x = self.positional_encoding(x)
        x = self.encoder(x)
        
        for transformer in self.transformers:
            x = transformer(x)
        
        x = self.decoder(x)
        return x

# LEGACY
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, config, input_size, hidden_size, window, horizon, num_heads=8, num_layers=1, attention_dropout=0.6, feedforward_dropout=0.1):
        super(SpatioTemporalTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window = window
        self.horizon = horizon
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        
        self.encoder = nn.Linear(self.input_size * window, self.hidden_size)
        self.positional_encoding = PositionalEncoding(self.input_size * window)
        
        self.transformers = nn.ModuleList([
            torch_geometric.nn.Sequential('x, mask', [
                (GraphWiseSelfAttention(config, self.hidden_size, self.hidden_size, window, horizon, num_heads, attention_dropout), 'x, mask -> x'),
                nn.LayerNorm(self.hidden_size),
                nn.Dropout(feedforward_dropout),
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(feedforward_dropout),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            ])
            for _ in range(num_layers)
        ])
        
        
        self.decoder = nn.Linear(self.hidden_size, self.hidden_size // window)
    
    def forward(self, x, edge_index, edge_weight):
        b, t, n, f = x.size()
        x = x.permute(0,2,1,3).reshape(b, n, t*f)
        
        x = self.encoder(x)
        x = self.positional_encoding(x)
        
        # x = x.permute(0,2,1,3)
        # x = x.reshape(b * n, t, f)
        # x = self.positional_encoding(x)
        # x = x.reshape(b, n, t, f)
        # x = x.permute(0,2,1,3)
        
        # create graph mask
        asj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([n, n]))
        mask = asj.to_dense() == 0
        
        for transformer in self.transformers:
            x = transformer(x, mask)
        
        x = self.decoder(x)
        return x