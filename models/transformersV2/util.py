import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the classic sinusoidal positional encoding.
    
    For an input tensor of shape (B, T, d_model) or (B, N, T, d_model),
    it adds a positional encoding to every token along the time axis.
    """
    def __init__(self, d_model, dropout=0.1, max_len=500):
        """
        Args:
            d_model (int): Dimensionality of the token embeddings.
            dropout (float): Dropout rate applied after adding PE.
            max_len (int): Maximum sequence length to precompute positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a (max_len, d_model) matrix; each row is the positional encoding for that time step.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the division term using the logarithm of 10000 (a common choice)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # pe is not a parameter, but persistent

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input embedding with shape either (B, T, d_model)
                              or (B, N, T, d_model).
        Returns:
            torch.Tensor: Output tensor after adding positional encodings.
        """
        if x.dim() == 3:
            # x has shape (B, T, d_model)
            x = x + self.pe[:, :x.size(1)]
        elif x.dim() == 4:
            # x has shape (B, N, T, d_model). Expand pe to (1,1,T,d_model) and add along the time dimension.
            x = x + self.pe[:, :x.size(2)].unsqueeze(1)
        else:
            raise ValueError("Unsupported input dimension for PositionalEncoding")
        return self.dropout(x)

def build_combined_mask(edge_index, edge_weight, N, T, mask_self=False):
    """
    Constructs a combined mask of shape (N*T, N*T) that embeds:
      - A temporal causal mask (T x T), and
      - A graph connectivity mask derived from edge_index and edge_weight.
    
    For every pair of nodes, if a connection exists (edge_weight > 0),
    the corresponding (T, T) block equals the temporal mask;
    otherwise, it is set to -inf.
    
    Arguments:
      edge_index: LongTensor of shape (2, E) where each column is (source, target).
      edge_weight: Tensor of shape (E,) with positive weights (e.g. 1).
      N: Number of nodes.
      T: Number of time steps.
      device: torch.device.
    
    Returns:
      A mask of shape (N*T, N*T) to be added to the attention scores.
    """
    # Build a dense connectivity indicator of shape (N, N)
    # For each edge (u->v): set A[v,u] = edge_weight (so only when there is an edge, we want to allow attention).
    A = torch.zeros((N, N)).to(edge_index.device)
    A[edge_index[1], edge_index[0]] = edge_weight  # Note: our convention: edge (u, v) means u -> v

    # Build a connectivity mask: if A > 0, we allow attention (0 added), else, we want to block by setting to -inf.
    connectivity_indicator = torch.where(A > 0, torch.zeros_like(A), torch.full_like(A, float('-inf')))
    
    if mask_self:
        # Mask self-attention: we do not want nodes to attend to themselves.
        connectivity_indicator.fill_diagonal_(float('-inf'))
    
    # Create a dense temporal (causal) mask of shape (T, T)
    # For example, we use a lower-triangular mask to prevent query at time t from attending to keys at future times
    temporal_mask = torch.triu(torch.ones((T, T)) * float('-inf'), diagonal=1).to(edge_index.device)
    
    """
    Now, we “lift” these masks to the full (N*T, N*T) mask.
    For the graph part, we create a block mask using a Kronecker product.
    The idea is:
       mask = kron(connectivity_indicator, ones(T, T)) + kron(ones(N, N), temporal_mask)
    For a connected node pair, connectivity_indicator==0 so that block becomes 0 + temporal_mask,
    and for an unconnected pair, connectivity_indicator==-inf so block remains -inf.
    """
    mask = torch.kron(connectivity_indicator, torch.ones((T, T), device=edge_index.device)) + torch.kron(torch.ones((N, N), device=edge_index.device), temporal_mask).to(edge_index.device)
    
    return mask.bool()  # shape: (N*T, N*T)