"""
Negative Sampler Module.

This module implements degree-biased negative sampling for skip-gram training.
Negative samples provide contrast - they tell the model which nodes should
NOT have similar embeddings (nodes that don't co-occur in walks).

Key Concept:
    Word2Vec (and node2vec) use degree-biased sampling where nodes are
    sampled proportionally to degree^0.75. This sublinear dampening:
    - Prevents high-degree nodes from dominating
    - Still gives reasonable coverage of hub nodes
    - Matches the original Word2Vec paper recommendations
"""

import torch
from typing import Optional, Tuple
import numpy as np


class DegreeBiasedNegativeSampler:
    """
    Degree-biased negative sampling with sublinear dampening.

    This sampler draws negative samples from a distribution proportional
    to degree^exponent, where the default exponent of 0.75 provides
    good balance between uniform and degree-proportional sampling.

    The samples are used as negative examples in training: pairs
    (target, negative) should have low similarity scores.

    Example:
        >>> import torch
        >>> from src.walks import DegreeBiasedNegativeSampler
        >>>
        >>> # Create edge index
        >>> edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
        >>> num_nodes = 4
        >>>
        >>> sampler = DegreeBiasedNegativeSampler(
        ...     edge_index=edge_index,
        ...     num_nodes=num_nodes,
        ...     exponent=0.75
        ... )
        >>>
        >>> # Sample 5 negatives for each of 100 positive pairs
        >>> negatives = sampler.sample(num_samples=100, num_negatives=5)
        >>> print(negatives.shape)  # [100, 5]
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        exponent: float = 0.75,
        device: Optional[torch.device] = None
    ):
        """
        Initialize negative sampler.

        Args:
            edge_index: Edge tensor of shape [2, num_edges]
            num_nodes: Total number of nodes
            exponent: Dampening exponent (default 0.75 from Word2Vec)
            device: Device to store probability tensor on
        """
        self.num_nodes = num_nodes
        self.exponent = exponent
        self.device = device or torch.device('cpu')

        # Compute degree for each node
        # We use out-degree (source side of edges)
        if edge_index.numel() == 0:
            degree = torch.ones(num_nodes, dtype=torch.float32)
        else:
            src = edge_index[0].cpu()
            degree = torch.zeros(num_nodes, dtype=torch.float32)
            for node in src.tolist():
                degree[node] += 1

            # Add small epsilon to avoid zero probability for isolated nodes
            degree = degree + 1e-8

        # Apply sublinear dampening: P(v) ∝ deg(v)^exponent
        # exponent < 1 reduces the influence of high-degree nodes
        sampling_weights = degree ** exponent

        # Normalize to get probabilities
        self.probs = sampling_weights / sampling_weights.sum()
        self.probs = self.probs.to(self.device)

        # Pre-compute for efficient sampling
        self._setup_alias_table()

    def _setup_alias_table(self) -> None:
        """
        Setup alias table for O(1) sampling.

        The alias method allows sampling from an arbitrary discrete
        distribution in O(1) time after O(n) preprocessing.
        """
        n = self.num_nodes
        probs_np = self.probs.cpu().numpy() * n

        # Initialize alias table
        self.alias = np.zeros(n, dtype=np.int64)
        self.prob_table = np.zeros(n, dtype=np.float32)

        # Separate into small and large probabilities
        smaller = []
        larger = []

        for i, p in enumerate(probs_np):
            if p < 1.0:
                smaller.append(i)
            else:
                larger.append(i)

        # Build alias table
        while smaller and larger:
            small_idx = smaller.pop()
            large_idx = larger.pop()

            self.prob_table[small_idx] = probs_np[small_idx]
            self.alias[small_idx] = large_idx

            probs_np[large_idx] = probs_np[large_idx] + probs_np[small_idx] - 1.0

            if probs_np[large_idx] < 1.0:
                smaller.append(large_idx)
            else:
                larger.append(large_idx)

        # Handle remaining elements
        while larger:
            large_idx = larger.pop()
            self.prob_table[large_idx] = 1.0

        while smaller:
            small_idx = smaller.pop()
            self.prob_table[small_idx] = 1.0

        # Convert to tensors
        self.alias = torch.from_numpy(self.alias).to(self.device)
        self.prob_table = torch.from_numpy(self.prob_table).to(self.device)

    def sample(
        self,
        num_samples: int,
        num_negatives: int = 5
    ) -> torch.Tensor:
        """
        Sample negative nodes.

        Uses the alias method for efficient sampling from the
        degree-biased distribution.

        Args:
            num_samples: Number of positive pairs (batch size)
            num_negatives: Number of negatives per positive

        Returns:
            Tensor of shape [num_samples, num_negatives] with node indices
        """
        total_samples = num_samples * num_negatives

        # Sample using alias method (O(1) per sample)
        # First, sample which bin
        idx = torch.randint(0, self.num_nodes, (total_samples,), device=self.device)

        # Then, decide whether to use original or alias
        u = torch.rand(total_samples, device=self.device)
        use_alias = u >= self.prob_table[idx]

        # Get final samples
        samples = torch.where(use_alias, self.alias[idx], idx)

        return samples.view(num_samples, num_negatives)

    def sample_multinomial(
        self,
        num_samples: int,
        num_negatives: int = 5
    ) -> torch.Tensor:
        """
        Sample using PyTorch's multinomial (simpler but O(n) per batch).

        This is a fallback method that's simpler but less efficient
        for large graphs.

        Args:
            num_samples: Number of positive pairs
            num_negatives: Number of negatives per positive

        Returns:
            Tensor of shape [num_samples, num_negatives]
        """
        total = num_samples * num_negatives
        samples = torch.multinomial(
            self.probs,
            total,
            replacement=True
        )
        return samples.view(num_samples, num_negatives)

    def sample_excluding(
        self,
        num_samples: int,
        num_negatives: int,
        exclude_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample negatives excluding specific nodes.

        This ensures negatives don't accidentally include the target
        or context nodes from positive pairs.

        Note: This is more expensive as it requires rejection sampling.

        Args:
            num_samples: Number of positive pairs
            num_negatives: Negatives per positive
            exclude_nodes: Tensor of shape [num_samples] with nodes to exclude

        Returns:
            Tensor of shape [num_samples, num_negatives]
        """
        # Sample with oversampling factor to account for rejections
        oversample_factor = 2
        candidates = self.sample(num_samples, num_negatives * oversample_factor)

        # Filter out excluded nodes (per row)
        result = torch.zeros(
            num_samples, num_negatives,
            dtype=torch.long, device=self.device
        )

        exclude_nodes = exclude_nodes.to(self.device)

        for i in range(num_samples):
            exclude = exclude_nodes[i].item()
            valid = candidates[i][candidates[i] != exclude]

            if len(valid) >= num_negatives:
                result[i] = valid[:num_negatives]
            else:
                # Not enough valid samples, just use what we have
                # and fill rest with re-samples
                result[i, :len(valid)] = valid
                extra = self.sample(1, num_negatives - len(valid)).squeeze(0)
                result[i, len(valid):] = extra

        return result

    def get_statistics(self) -> dict:
        """
        Get statistics about the sampling distribution.

        Returns:
            Dictionary with sampling statistics
        """
        probs_np = self.probs.cpu().numpy()

        return {
            'num_nodes': self.num_nodes,
            'exponent': self.exponent,
            'max_prob': float(probs_np.max()),
            'min_prob': float(probs_np.min()),
            'entropy': float(-np.sum(probs_np * np.log(probs_np + 1e-10))),
            'effective_samples': float(1.0 / np.sum(probs_np ** 2)),
            'gini_coefficient': float(
                np.sum(np.abs(probs_np[:, None] - probs_np[None, :])) /
                (2 * self.num_nodes * np.sum(probs_np))
            )
        }


class UniformNegativeSampler:
    """
    Uniform negative sampling (no degree bias).

    This is simpler but may not work as well for graphs with
    skewed degree distributions.
    """

    def __init__(self, num_nodes: int, device: Optional[torch.device] = None):
        """
        Initialize uniform sampler.

        Args:
            num_nodes: Total number of nodes
            device: Device for output tensors
        """
        self.num_nodes = num_nodes
        self.device = device or torch.device('cpu')

    def sample(
        self,
        num_samples: int,
        num_negatives: int = 5
    ) -> torch.Tensor:
        """
        Sample uniformly at random.

        Args:
            num_samples: Number of positive pairs
            num_negatives: Negatives per positive

        Returns:
            Tensor of shape [num_samples, num_negatives]
        """
        return torch.randint(
            0, self.num_nodes,
            (num_samples, num_negatives),
            device=self.device
        )
