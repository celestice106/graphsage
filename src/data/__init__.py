"""
Data Module for GraphSAGE Training.

This module handles:
1. Loading graphs from Memory Bank system
2. Extracting memory-only view for GraphSAGE
3. Computing node features
4. Creating PyTorch datasets

Classes:
    GraphLoader: Load full graph from Memory Bank storage
    GraphSAGEViewExtractor: Extract memory-only subgraph
    MemoryFeatureExtractor: Compute 7-dim features for memory nodes
    GraphSAGEDataset: PyTorch dataset wrapper

Example:
    >>> from src.data import GraphSAGEViewExtractor, MemoryFeatureExtractor
    >>>
    >>> # Extract view from full graph
    >>> extractor = GraphSAGEViewExtractor(full_graph)
    >>> memory_ids, edge_index, node_mapping = extractor.extract()
    >>>
    >>> # Compute features
    >>> feature_extractor = MemoryFeatureExtractor(full_graph, node_mapping)
    >>> features = feature_extractor.extract(edge_index)
"""

from .graph_loader import GraphLoader, MockGraphStore
from .view_extractor import GraphSAGEViewExtractor
from .feature_extractor import MemoryFeatureExtractor
from .dataset import GraphSAGEDataset, create_data_object

__all__ = [
    'GraphLoader',
    'MockGraphStore',
    'GraphSAGEViewExtractor',
    'MemoryFeatureExtractor',
    'GraphSAGEDataset',
    'create_data_object',
]
