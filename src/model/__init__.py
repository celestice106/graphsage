"""
Model Module for GraphSAGE.

This module implements the GraphSAGE model architecture and loss function
for learning structural embeddings.

Components:
    - layers.py: Custom layer wrappers and utilities
    - graphsage.py: Main ProductionGraphSAGE model
    - loss.py: Skip-gram loss with negative sampling

The model takes node features and graph structure, then produces
L2-normalized embeddings suitable for dot-product similarity.

Example:
    >>> from src.model import ProductionGraphSAGE, SkipGramLoss
    >>>
    >>> # Create model
    >>> model = ProductionGraphSAGE(
    ...     in_channels=7,
    ...     hidden_channels=64,
    ...     out_channels=64,
    ...     dropout=0.3
    ... )
    >>>
    >>> # Forward pass
    >>> embeddings = model(features, edge_index)
    >>>
    >>> # Compute loss
    >>> loss_fn = SkipGramLoss()
    >>> loss = loss_fn(embeddings, targets, contexts, negatives)
"""

from .graphsage import ProductionGraphSAGE
from .loss import SkipGramLoss

__all__ = [
    'ProductionGraphSAGE',
    'SkipGramLoss',
]
