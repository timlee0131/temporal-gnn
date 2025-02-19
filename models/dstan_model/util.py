import torch
import torch.nn as nn
import math

def rope(x, base=10000):
    B, T, N, F = x.size()
    
    assert F % 2 == 0, "Feature dimension must be even for ROPE"
    
    half_dim = F // 2
    
    # Compute the position indices for time steps: shape (T,)
    pos = torch.arange(T, device=x.device, dtype=x.dtype)
    
    # Compute inverse frequency vector for the half-dim (shape: (half_dim,))
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, device=x.device, dtype=x.dtype) / half_dim))
    
    # Outer product: (T, half_dim)  = (T, 1) * (1, half_dim)
    freqs = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
    
    # Compute cosine and sine factors: shape (T, half_dim)
    cos_pos = torch.cos(freqs)  # (T, half_dim)
    sin_pos = torch.sin(freqs)  # (T, half_dim)
    
    # These need to be broadcast to the input shape.
    # Expand so that they can be added to x along T and broadcast over B and N.
    # We want shape (1, T, 1, half_dim)
    cos_pos = cos_pos.unsqueeze(0).unsqueeze(2)
    sin_pos = sin_pos.unsqueeze(0).unsqueeze(2)
    
    # Split x into two halves along the feature dimension.
    x1 = x[..., :half_dim]  # shape: (B, T, N, half_dim)
    x2 = x[..., half_dim:]  # shape: (B, T, N, half_dim)
    
    # Apply rotary transformation:
    # out_first = x1 * cos - x2 * sin
    # out_second = x1 * sin + x2 * cos
    out_first  = x1 * cos_pos - x2 * sin_pos
    out_second = x1 * sin_pos + x2 * cos_pos
    
    # Concatenate along the feature dimension to get back to shape (B, T, N, F)
    out = torch.cat([out_first, out_second], dim=-1)
    
    return out

def pe_mem_future(data, future_query, base=10000):
    """
    Applies the classical sinusoidal positional encoding (from the Transformer paper)
    to both the original data and the future query in a continuous fashion.

    class PositionalEncoding(nn.Module):
    The original data gets positions 0, 1, 2, …, T_data-1 and the future query
    gets positions T_data, T_data+1, …, T_data+T_future-1, ensuring that the model
    sees a continuous progression of time steps.

    Parameters:
    data (Tensor): Original data of shape (B, T_data, N, F)
    future_query (Tensor): Future query of shape (B, T_future, N, F)
    base (float): Base constant for positional encoding (default: 10000)
    
    Returns:
    encoded_data (Tensor): Data with positional encoding added (B, T_data, N, F)
    encoded_future (Tensor): Future query with positional encoding added (B, T_future, N, F)
    """
    B, T_data, N, F = data.size()
    B_f, T_future, N_f, F_f = future_query.size()
    assert B == B_f and N == N_f and F == F_f, "Data and future_query must agree in batch, nodes, and feature dims."

    # Compute position indices for data: positions 0, 1, 2, ... T_data-1
    pos_data = torch.arange(0, T_data, dtype=data.dtype, device=data.device).unsqueeze(1)  # (T_data, 1)
    # For future queries, positions start where data left off.
    pos_future = torch.arange(T_data, T_data + T_future, dtype=data.dtype, device=data.device).unsqueeze(1)  # (T_future, 1)

    # Create the divisor term; note we use only half the feature dimension as in the Transformer paper.
    # (i.e. different dimensions get paired: even => sin, odd => cos)
    div_term = torch.exp(torch.arange(0, F, 2, dtype=data.dtype, device=data.device) *
                        -(math.log(base) / F))  # shape: (F/2,)

    # Compute the sinusoidal positional encodings for the data.
    pe_data = torch.zeros(T_data, F, dtype=data.dtype, device=data.device)
    pe_data[:, 0::2] = torch.sin(pos_data * div_term)
    pe_data[:, 1::2] = torch.cos(pos_data * div_term)

    # Compute the positional encoding for the future query.
    pe_future = torch.zeros(T_future, F, dtype=data.dtype, device=data.device)
    pe_future[:, 0::2] = torch.sin(pos_future * div_term)
    pe_future[:, 1::2] = torch.cos(pos_future * div_term)

    # Reshape for addition: we want dimensions (1, T, 1, F) to broadcast correctly
    pe_data = pe_data.unsqueeze(0).unsqueeze(2)     # (1, T_data, 1, F)
    pe_future = pe_future.unsqueeze(0).unsqueeze(2)   # (1, T_future, 1, F)

    # Add the positional encoding to the inputs
    encoded_data = data + pe_data
    encoded_future = future_query + pe_future

    return encoded_data, encoded_future

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

def build_sparse_combined_mask(edge_index, edge_weight, N, T, mask_self=False):
    """
    Constructs a sparse combined mask of shape (N*T, N*T) that encodes:
      - A temporal causal mask (only allowing positions where query time >= key time)
      - A graph connectivity mask derived from edge_index and edge_weight.
    
    For every valid edge (u->v), where by convention edge_index[0] is u (source)
    and edge_index[1] is v (target), only the block corresponding to node pair (v,u)
    will allow positions (i, j) (with 0 <= i, j < T) such that i >= j. All other entries
    are implicitly not stored in the sparse mask.
    
    If mask_self==True, self connections (where u==v) are removed (i.e. no allowed block).
    
    Args:
      edge_index: LongTensor of shape (2, E) on some device (e.g. GPU) with each column (u, v).
      edge_weight: Tensor of shape (E,) with positive weights.
      N: Number of nodes.
      T: Number of time steps.
      mask_self: If True, disallow self-attention (i.e. skip edges where u == v).
    
    Returns:
      sparse_mask: A torch.sparse_coo_tensor of shape (N*T, N*T) containing 0 for allowed positions.
                   (Your attention code should treat missing indices as -inf.)
    """
    device = edge_index.device

    # Optionally filter out self connections.
    if mask_self:
        valid = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, valid]
        edge_weight = edge_weight[valid]
        
    E = edge_index.shape[1]  # number of valid edges

    # Get allowed temporal indices within a block of size T x T: we allow (i,j) only if i >= j.
    tril = torch.tril_indices(T, T, device=device)  # shape: (2, L)
    L = tril.shape[1]  # number of allowed indices per block

    # For each valid edge, compute the global row and column indices.
    # For an edge from u to v (u: source, v: target): the block is located at rows v*T : (v+1)*T and columns u*T : (u+1)*T.
    # Then the allowed positions are:
    #   row index = v * T + tril[0]
    #   col index = u * T + tril[1]
    row_idx = edge_index[1].unsqueeze(1) * T + tril[0].unsqueeze(0)  # shape (E, L)
    col_idx = edge_index[0].unsqueeze(1) * T + tril[1].unsqueeze(0)  # shape (E, L)

    # Flatten indices
    row_idx = row_idx.reshape(-1)
    col_idx = col_idx.reshape(-1)
    indices = torch.stack([row_idx, col_idx], dim=0)  # shape (2, E * L)

    # For each allowed position, we assign a value of 0 (i.e. no masking penalty).
    values = torch.zeros(E * L, device=device)

    # Build and return the sparse mask.
    sparse_mask = torch.sparse_coo_tensor(indices, values, size=(N*T, N*T), device=device)
    return sparse_mask
