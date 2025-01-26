import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_heads: int):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        
        self.W_o = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # x: [batch, time, nodes, features]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        q = q.view(q.size(0), q.size(1), q.size(2), self.n_heads, self.hidden_size // self.n_heads)
        k = k.view(k.size(0), k.size(1), k.size(2), self.n_heads, self.hidden_size // self.n_heads)
        v = v.view(v.size(0), v.size(1), v.size(2), self.n_heads, self.hidden_size // self.n_heads)
        
        q = q.permute(0, 3, 1, 2, 4).contiguous().view(-1, q.size(1), q.size(2), self.hidden_size // self.n_heads)
        k = k.permute(0, 3, 1, 2, 4).contiguous().view(-1, k.size(1), k.size(2), self.hidden_size // self.n_heads)
        v = v.permute(0, 3, 1, 2, 4).contiguous().view(-1, v.size(1), v.size(2), self.hidden_size // self.n_heads)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size // self.n_heads) ** 0.5
        attention = F.softmax(attention, dim=-1)
        
        x = torch.matmul(attention, v)
        x = x.view(x.size(0), x.size(1), x.size(2), self.n_heads * (self.hidden_size // self.n_heads))
        
        x = self.W_o(x)
        
        return x