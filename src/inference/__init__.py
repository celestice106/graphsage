"""
Inference Module for Production Deployment.

This module provides production-ready inference components:
- Efficient encoder for computing embeddings
- Caching with dirty tracking for dynamic graphs
- Integration interface for Memory Bank

Components:
    StructuralEncoder: Main encoder 
    EmbeddingCache: Cache with invalidation support

Example:
    >>> from src.inference import StructuralEncoder
    >>>
    >>> encoder = StructuralEncoder(
    ...     model_path='exports/graphsage.pt',
    ...     device='cuda'
    ... )
    >>>
    >>> # Get embeddings for all nodes
    >>> embeddings = encoder.encode_all(full_graph)
    >>>
    >>> # Get embedding for single node
    >>> emb = encoder.encode_single(node_idx, full_graph)
"""

from .encoder import StructuralEncoder
from .cache import EmbeddingCache

__all__ = [
    'StructuralEncoder',
    'EmbeddingCache',
]
