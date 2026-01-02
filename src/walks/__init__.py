"""
Random Walk Module for GraphSAGE Training.

This module implements the random walk co-occurrence approach (GDS-style)
for generating training data. It includes:

1. Random walk generation on the graph
2. Co-occurrence pair extraction (skip-gram style)
3. Degree-biased negative sampling

The core idea: nodes that appear together in random walks are structurally
similar and should have similar embeddings.

Classes:
    RandomWalkGenerator: Generate random walks from each node
    CooccurrencePairSampler: Extract (target, context) pairs from walks
    DegreeBiasedNegativeSampler: Sample negative nodes for contrast

Example:
    >>> from src.walks import RandomWalkGenerator, CooccurrencePairSampler
    >>> from src.walks import DegreeBiasedNegativeSampler
    >>>
    >>> # Generate walks
    >>> walker = RandomWalkGenerator(edge_index, num_nodes)
    >>> walks = walker.generate_all_walks()
    >>>
    >>> # Extract pairs
    >>> sampler = CooccurrencePairSampler(context_window=10)
    >>> pairs = sampler.extract_pairs(walks)
    >>>
    >>> # Setup negative sampler
    >>> neg_sampler = DegreeBiasedNegativeSampler(edge_index, num_nodes)
"""

from .generator import RandomWalkGenerator
from .pair_sampler import CooccurrencePairSampler
from .negative_sampler import DegreeBiasedNegativeSampler

__all__ = [
    'RandomWalkGenerator',
    'CooccurrencePairSampler',
    'DegreeBiasedNegativeSampler',
]
