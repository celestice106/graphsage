"""
Co-occurrence Pair Sampler Module.

This module extracts (target, context) pairs from random walks for
skip-gram training. The core idea is that nodes appearing close together
in walks are structurally similar.

Key Concept:
    Given a walk [n1, n2, n3, n4, n5] and window size 2:
    - Target n3 has contexts: n1, n2, n4, n5
    - This creates pairs: (n3, n1), (n3, n2), (n3, n4), (n3, n5)

    These pairs define what "co-occurring" means in the graph.
"""

import torch
from typing import List, Tuple, Optional
from collections import Counter
import numpy as np


class CooccurrencePairSampler:
    """
    Extract (target, context) pairs from random walks.

    This implements the skip-gram pair extraction used in Word2Vec,
    but for graph nodes instead of words. Nodes within a context window
    of each other in walks are considered co-occurring.

    The extracted pairs are used as positive examples in training:
    the model learns that co-occurring nodes should have similar embeddings.

    Example:
        >>> from src.walks import CooccurrencePairSampler
        >>>
        >>> # Example walks
        >>> walks = [[0, 1, 2, 3], [1, 2, 0, 3]]
        >>>
        >>> sampler = CooccurrencePairSampler(context_window=2)
        >>> pairs = sampler.extract_pairs(walks)
        >>>
        >>> # pairs contains (target, context) tuples
        >>> print(f"Extracted {len(pairs)} pairs")
    """

    def __init__(self, context_window: int = 10):
        """
        Initialize pair sampler.

        Args:
            context_window: Size of context window on each side.
                           Total context = 2 * context_window nodes.
        """
        self.context_window = context_window

    def extract_pairs(self, walks: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Extract all positive pairs from walks.

        For each position in each walk, we pair the target node with
        all context nodes within the window.

        Args:
            walks: List of walks (each walk is a list of node indices)

        Returns:
            List of (target, context) tuples
        """
        pairs = []

        for walk in walks:
            walk_len = len(walk)

            for i, target in enumerate(walk):
                # Define context window boundaries
                # Window extends context_window positions in each direction
                start = max(0, i - self.context_window)
                end = min(walk_len, i + self.context_window + 1)

                # Add pairs for all context nodes
                for j in range(start, end):
                    if i != j:  # Don't pair node with itself
                        context = walk[j]
                        pairs.append((target, context))

        return pairs

    def extract_pairs_tensor(
        self,
        walks: List[List[int]],
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pairs and return as tensors.

        This is more efficient for training as it avoids repeated
        list-to-tensor conversions.

        Args:
            walks: List of walks
            device: Device to place tensors on

        Returns:
            Tuple of (targets, contexts) tensors, each of shape [num_pairs]
        """
        pairs = self.extract_pairs(walks)

        if not pairs:
            # Return empty tensors if no pairs
            targets = torch.tensor([], dtype=torch.long)
            contexts = torch.tensor([], dtype=torch.long)
        else:
            targets = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        if device is not None:
            targets = targets.to(device)
            contexts = contexts.to(device)

        return targets, contexts

    def extract_pairs_with_counts(
        self,
        walks: List[List[int]]
    ) -> Tuple[List[Tuple[int, int]], Counter]:
        """
        Extract pairs and count co-occurrence frequencies.

        The counts can be used for weighted training or analysis.

        Args:
            walks: List of walks

        Returns:
            Tuple of (unique pairs list, counter with frequencies)
        """
        pairs = self.extract_pairs(walks)
        pair_counts = Counter(pairs)

        unique_pairs = list(pair_counts.keys())

        return unique_pairs, pair_counts

    def get_statistics(self, walks: List[List[int]]) -> dict:
        """
        Get statistics about extracted pairs.

        Args:
            walks: List of walks

        Returns:
            Dictionary with pair statistics
        """
        pairs = self.extract_pairs(walks)

        if not pairs:
            return {
                'num_pairs': 0,
                'unique_pairs': 0,
                'unique_targets': 0,
                'unique_contexts': 0,
                'avg_contexts_per_target': 0,
                'context_window': self.context_window
            }

        targets = [p[0] for p in pairs]
        contexts = [p[1] for p in pairs]

        target_counts = Counter(targets)

        return {
            'num_pairs': len(pairs),
            'unique_pairs': len(set(pairs)),
            'unique_targets': len(set(targets)),
            'unique_contexts': len(set(contexts)),
            'avg_contexts_per_target': np.mean(list(target_counts.values())),
            'max_contexts_per_target': max(target_counts.values()),
            'min_contexts_per_target': min(target_counts.values()),
            'context_window': self.context_window
        }


class SubsampledPairSampler(CooccurrencePairSampler):
    """
    Pair sampler with frequency-based subsampling.

    High-frequency nodes (appearing in many walks) can dominate training.
    This sampler uses Word2Vec-style subsampling to reduce their influence.

    The probability of keeping a pair involving node n is:
        P(keep) = sqrt(t / freq(n)) + t / freq(n)

    where freq(n) is the node's frequency and t is a threshold.
    """

    def __init__(
        self,
        context_window: int = 10,
        subsample_threshold: float = 1e-5,
        min_count: int = 1
    ):
        """
        Initialize subsampled pair sampler.

        Args:
            context_window: Size of context window
            subsample_threshold: Frequency threshold for subsampling
            min_count: Minimum frequency to include a node
        """
        super().__init__(context_window)
        self.subsample_threshold = subsample_threshold
        self.min_count = min_count

    def extract_pairs(
        self,
        walks: List[List[int]],
        seed: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Extract pairs with frequency-based subsampling.

        Args:
            walks: List of walks
            seed: Random seed for subsampling

        Returns:
            List of (target, context) pairs after subsampling
        """
        if seed is not None:
            np.random.seed(seed)

        # Count node frequencies across all walks
        node_counts = Counter()
        for walk in walks:
            node_counts.update(walk)

        total_count = sum(node_counts.values())

        # Compute keep probabilities
        keep_prob = {}
        for node, count in node_counts.items():
            if count < self.min_count:
                keep_prob[node] = 0.0
            else:
                freq = count / total_count
                # Word2Vec subsampling formula
                prob = (np.sqrt(freq / self.subsample_threshold) +
                       self.subsample_threshold / freq)
                keep_prob[node] = min(1.0, prob)

        # Extract pairs with subsampling
        pairs = []

        for walk in walks:
            walk_len = len(walk)

            for i, target in enumerate(walk):
                # Subsample target
                if np.random.random() > keep_prob.get(target, 1.0):
                    continue

                start = max(0, i - self.context_window)
                end = min(walk_len, i + self.context_window + 1)

                for j in range(start, end):
                    if i != j:
                        context = walk[j]
                        # Subsample context
                        if np.random.random() <= keep_prob.get(context, 1.0):
                            pairs.append((target, context))

        return pairs


def estimate_pair_count(
    num_nodes: int,
    walks_per_node: int,
    walk_length: int,
    context_window: int,
    avg_walk_length: Optional[float] = None
) -> int:
    """
    Estimate the number of pairs that will be generated.

    This is useful for pre-allocating memory or estimating training time.

    Args:
        num_nodes: Number of nodes in graph
        walks_per_node: Walks per node
        walk_length: Maximum walk length
        context_window: Context window size
        avg_walk_length: Average actual walk length (if known)

    Returns:
        Estimated number of pairs
    """
    if avg_walk_length is None:
        # Assume walks complete (optimistic estimate)
        avg_walk_length = walk_length

    total_walks = num_nodes * walks_per_node

    # For each position in a walk, we get ~2*context_window pairs
    # (context_window on each side)
    pairs_per_walk = avg_walk_length * min(2 * context_window, avg_walk_length - 1)

    return int(total_walks * pairs_per_walk)
