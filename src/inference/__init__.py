"""
Inference Module for Production Deployment.

This module provides production-ready inference components:
- Efficient encoder for computing embeddings
- Caching with dirty tracking for dynamic graphs
- Integration interface for Memory R1

Components:
    MemoryR1StructuralEncoder: Main encoder for Memory R1 integration
    EmbeddingCache: Cache with invalidation support

Example:
    >>> from src.inference import MemoryR1StructuralEncoder
    >>>
    >>> encoder = MemoryR1StructuralEncoder(
    ...     model_path='exports/graphsage_production.pt',
    ...     device='cuda'
    ... )
    >>>
    >>> # Get embeddings for all nodes
    >>> embeddings = encoder.encode_all(edge_index, num_nodes, full_graph)
    >>>
    >>> # Get embedding for single node
    >>> emb = encoder.encode_single(node_idx, edge_index, num_nodes, full_graph)
"""

from .encoder import MemoryR1StructuralEncoder
from .cache import EmbeddingCache

__all__ = [
    'MemoryR1StructuralEncoder',
    'EmbeddingCache',
]
