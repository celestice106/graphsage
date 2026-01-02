"""
Utilities Module.

This module provides helper functions for:
- Graph manipulation and analysis
- Evaluation metrics
- Embedding visualization

Components:
    graph_utils: Graph analysis and manipulation
    metrics: Evaluation metrics for embeddings
    visualization: Plotting and visualization tools
"""

from .graph_utils import (
    compute_degree_distribution,
    compute_graph_statistics,
    sample_subgraph
)
from .metrics import (
    compute_neighbor_similarity,
    evaluate_link_prediction,
    compute_embedding_statistics
)
from .visualization import (
    plot_training_curves,
    plot_embeddings_tsne,
    plot_degree_distribution
)

__all__ = [
    # Graph utils
    'compute_degree_distribution',
    'compute_graph_statistics',
    'sample_subgraph',
    # Metrics
    'compute_neighbor_similarity',
    'evaluate_link_prediction',
    'compute_embedding_statistics',
    # Visualization
    'plot_training_curves',
    'plot_embeddings_tsne',
    'plot_degree_distribution',
]
