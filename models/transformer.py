import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import PositionalEncoding, SelfAttention

class TemporalTransformer(nn.Module):
    def __init__(self, config, input_size, hidden_size, window, horizon, num_heads=8, num_layers=1, attention_dropout=0.6, feedforward_dropout=0.1):
        super(TemporalTransformer, self).__init__()
        
        self.input_size = input_size * window
        self.hidden_plus_window_size = hidden_size * window
        self.window = window
        self.horizon = horizon
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        
        self.encoder = nn.Linear(self.input_size, self.hidden_plus_window_size)
        self.positional_encoding = PositionalEncoding(self.hidden_plus_window_size)
        self.transformers = nn.ModuleList([
            nn.Sequential(
                SelfAttention(config, self.hidden_plus_window_size, self.hidden_plus_window_size, window, horizon, num_heads, attention_dropout),
                nn.LayerNorm(self.hidden_plus_window_size),
                nn.Dropout(feedforward_dropout),
                nn.Linear(self.hidden_plus_window_size, self.hidden_plus_window_size * 2),
                nn.ReLU(),
                nn.Dropout(feedforward_dropout),
                nn.Linear(self.hidden_plus_window_size * 2, self.hidden_plus_window_size),
                nn.LayerNorm(self.hidden_plus_window_size)
            )
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Linear(self.hidden_plus_window_size, hidden_size)
    
    def forward(self, x):
        b, t, n, f = x.size()
        x = x.permute(0, 2, 1, 3).reshape(b, n, t * f)
        
        x = self.encoder(x)
        x = self.positional_encoding(x)
        
        for transformer in self.transformers:
            x = transformer(x)
        
        x = self.decoder(x)
        
        return x