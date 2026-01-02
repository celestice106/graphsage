"""
Graph Utilities Module.

This module provides helper functions for graph analysis and manipulation.
"""

import torch
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter


def compute_degree_distribution(edge_index: torch.Tensor, num_nodes: int) -> Dict:
    """
    Compute degree distribution statistics.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Total number of nodes

    Returns:
        Dictionary with degree statistics
    """
    if edge_index.numel() == 0:
        return {
            'degrees': torch.zeros(num_nodes, dtype=torch.long),
            'mean': 0.0,
            'std': 0.0,
            'min': 0,
            'max': 0,
            'histogram': {}
        }

    # Compute degree (treating as undirected by counting both directions)
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()

    out_degrees = torch.bincount(src, minlength=num_nodes)
    in_degrees = torch.bincount(dst, minlength=num_nodes)
    total_degrees = out_degrees + in_degrees

    # Statistics
    degrees_float = total_degrees.float()

    # Histogram
    degree_counts = Counter(total_degrees.tolist())
    histogram = {int(k): v for k, v in sorted(degree_counts.items())}

    return {
        'degrees': total_degrees,
        'mean': degrees_float.mean().item(),
        'std': degrees_float.std().item(),
        'min': int(total_degrees.min().item()),
        'max': int(total_degrees.max().item()),
        'histogram': histogram,
        'isolated_nodes': int((total_degrees == 0).sum().item())
    }


def compute_graph_statistics(edge_index: torch.Tensor, num_nodes: int) -> Dict:
    """
    Compute comprehensive graph statistics.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Total number of nodes

    Returns:
        Dictionary with graph statistics
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0

    degree_stats = compute_degree_distribution(edge_index, num_nodes)

    # Density
    max_edges = num_nodes * (num_nodes - 1)  # Directed graph
    density = num_edges / max_edges if max_edges > 0 else 0

    # Connected components (approximation via BFS from random nodes)
    # For proper implementation, use NetworkX
    connected_components = None  # Would need full BFS

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': degree_stats['mean'],
        'max_degree': degree_stats['max'],
        'min_degree': degree_stats['min'],
        'isolated_nodes': degree_stats['isolated_nodes'],
        'degree_histogram': degree_stats['histogram']
    }


def sample_subgraph(
    edge_index: torch.Tensor,
    num_nodes: int,
    sample_size: int,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    """
    Sample a random subgraph.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Total number of nodes
        sample_size: Number of nodes to sample
        seed: Random seed

    Returns:
        Tuple of (new_edge_index, sampled_node_indices, mapping)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Sample nodes
    sample_size = min(sample_size, num_nodes)
    sampled_indices = torch.randperm(num_nodes)[:sample_size]

    # Create mapping
    mapping = {int(old): new for new, old in enumerate(sampled_indices.tolist())}
    sampled_set = set(mapping.keys())

    # Filter edges
    if edge_index.numel() == 0:
        new_edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()

        new_edges = []
        for s, d in zip(src, dst):
            if s in sampled_set and d in sampled_set:
                new_edges.append([mapping[s], mapping[d]])

        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
        else:
            new_edge_index = torch.zeros((2, 0), dtype=torch.long)

    return new_edge_index, sampled_indices, mapping


def get_neighbors(
    node: int,
    edge_index: torch.Tensor,
    num_hops: int = 1
) -> List[int]:
    """
    Get neighbors of a node up to k hops.

    Args:
        node: Node index
        edge_index: Edge indices
        num_hops: Number of hops

    Returns:
        List of neighbor node indices
    """
    if edge_index.numel() == 0:
        return []

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    # Build adjacency list
    adj = {}
    for s, d in zip(src, dst):
        if s not in adj:
            adj[s] = []
        adj[s].append(d)

    # BFS for k hops
    visited = {node}
    current_frontier = {node}

    for _ in range(num_hops):
        next_frontier = set()
        for n in current_frontier:
            for neighbor in adj.get(n, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        current_frontier = next_frontier

    visited.remove(node)  # Remove the starting node
    return list(visited)


def compute_edge_density(edge_index: torch.Tensor, num_nodes: int) -> float:
    """
    Compute edge density of the graph.

    Args:
        edge_index: Edge indices
        num_nodes: Number of nodes

    Returns:
        Edge density (0 to 1)
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    max_edges = num_nodes * (num_nodes - 1)
    return num_edges / max_edges if max_edges > 0 else 0.0


def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Convert directed edge_index to undirected.

    Args:
        edge_index: Directed edge indices [2, num_edges]

    Returns:
        Undirected edge indices [2, 2*num_edges] (with duplicates removed)
    """
    if edge_index.numel() == 0:
        return edge_index

    # Add reverse edges
    reverse_edges = edge_index.flip(0)
    all_edges = torch.cat([edge_index, reverse_edges], dim=1)

    # Remove duplicates
    all_edges = torch.unique(all_edges, dim=1)

    return all_edges
