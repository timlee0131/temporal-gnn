import numpy as np
import torch
from torch_geometric.data import Data

def create_ramanujan_expander(num_nodes, d):
    """
    Creates a d-regular graph and attempts to optimize it to have 
    Ramanujan properties by iteratively improving the spectral gap.
    
    Args:
        num_nodes: Number of nodes in the graph
        d: Degree of each node (must be even for d-regular graphs)
    
    Returns:
        edge_index: Tensor of shape [2, num_edges] containing the edge indices
    """
    if d % 2 != 0:
        raise ValueError("For d-regular graphs, d must be even")
    
    if d >= num_nodes:
        raise ValueError("d must be less than the number of nodes")
    
    # Initialize with a random d-regular graph
    stubs = torch.arange(num_nodes).repeat_interleave(d)
    stubs = stubs[torch.randperm(stubs.size(0))]
    
    edges = []
    for i in range(0, stubs.size(0), 2):
        u, v = stubs[i].item(), stubs[i+1].item()
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.append((u, v))
            edges.append((v, u))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Optimize the graph to have better Ramanujan properties
    best_edge_index = edge_index.clone()
    best_lambda2 = float('inf')
    ramanujan_bound = 2 * np.sqrt(d - 1)
    
    # Simple optimization: Try several random configurations and keep the best
    for _ in range(10):  # Try 10 different configurations
        # Create a new random d-regular graph
        stubs = torch.arange(num_nodes).repeat_interleave(d)
        stubs = stubs[torch.randperm(stubs.size(0))]
        
        edges = []
        for i in range(0, stubs.size(0), 2):
            u, v = stubs[i].item(), stubs[i+1].item()
            if u != v and (u, v) not in edges and (v, u) not in edges:
                edges.append((u, v))
                edges.append((v, u))
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Check spectral properties
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for i, j in edge_index.t():
            adj_matrix[i, j] = 1
        
        eigenvalues = torch.linalg.eigvalsh(adj_matrix)
        eigenvalues = torch.sort(torch.abs(eigenvalues))[0]
        lambda2 = eigenvalues[-2].item()
        
        # Keep the best configuration
        if lambda2 < best_lambda2:
            best_lambda2 = lambda2
            best_edge_index = edge_index.clone()
            
            # If we found a Ramanujan graph, we can stop
            if lambda2 <= ramanujan_bound:
                break
    
    # Create a PyTorch Geometric Data object
    data = Data(x=torch.ones(num_nodes, 1), edge_index=best_edge_index)
    
    is_ramanujan = best_lambda2 <= ramanujan_bound
    print(f"Created {'Ramanujan' if is_ramanujan else 'non-Ramanujan'} graph with λ₂ = {best_lambda2:.4f} (bound: {ramanujan_bound:.4f})")
    
    return best_edge_index
