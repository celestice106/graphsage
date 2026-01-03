"""
GraphSAGE Training Module for Memory Bank Structural Embeddings.

This package implements a complete GraphSAGE training pipeline using the
random walk co-occurrence approach (GDS-style) for learning structural
embeddings of memory nodes.

Submodules:
    - data: Graph loading, view extraction, and feature computation
    - walks: Random walk generation and pair sampling
    - model: GraphSAGE architecture and loss functions
    - training: Training pipeline with early stopping and checkpointing
    - inference: Encoder with caching
    - utils: Helper functions for metrics, visualization, and graph operations

Example:
    >>> from src.data import GraphSAGEViewExtractor, MemoryFeatureExtractor
    >>> from src.walks import RandomWalkGenerator, CooccurrencePairSampler
    >>> from src.model import GraphSAGE
    >>> from src.training import GraphSAGETrainer
"""

__version__ = "1.0.0"

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}
