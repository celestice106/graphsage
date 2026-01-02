"""
Batch Generator Module.

This module handles the creation of training batches from positive pairs.
It shuffles data, chunks into batches, and samples negatives on-the-fly.
"""

import torch
from typing import List, Tuple, Iterator, Optional
from dataclasses import dataclass
import random
import numpy as np


@dataclass
class TrainingData:
    """
    Container for all training data.

    Attributes:
        positive_pairs: List of (target, context) tuples
        features: Node features tensor [num_nodes, num_features]
        edge_index: Edge indices tensor [2, num_edges]
        num_nodes: Total number of nodes
    """
    positive_pairs: List[Tuple[int, int]]
    features: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int

    def __post_init__(self):
        """Validate data after initialization."""
        assert len(self.positive_pairs) > 0, "Need at least one positive pair"
        assert self.features.shape[0] == self.num_nodes, \
            f"Features shape {self.features.shape[0]} != num_nodes {self.num_nodes}"

    def to(self, device: torch.device) -> 'TrainingData':
        """Move tensors to device."""
        return TrainingData(
            positive_pairs=self.positive_pairs,
            features=self.features.to(device),
            edge_index=self.edge_index.to(device),
            num_nodes=self.num_nodes
        )


@dataclass
class Batch:
    """
    A single training batch.

    Attributes:
        targets: Target node indices [batch_size]
        contexts: Context node indices [batch_size]
        negatives: Negative node indices [batch_size, num_negatives]
        size: Number of pairs in batch
    """
    targets: torch.Tensor
    contexts: torch.Tensor
    negatives: torch.Tensor
    size: int

    def to(self, device: torch.device) -> 'Batch':
        """Move all tensors to device."""
        return Batch(
            targets=self.targets.to(device),
            contexts=self.contexts.to(device),
            negatives=self.negatives.to(device),
            size=self.size
        )


class BatchGenerator:
    """
    Generate training batches from positive pairs.

    This generator:
    1. Shuffles positive pairs at the start of each epoch
    2. Chunks pairs into fixed-size batches
    3. Samples negatives on-the-fly for each batch

    The on-the-fly negative sampling is more memory efficient than
    pre-computing all negatives and provides fresh negatives each epoch.

    Example:
        >>> from src.training import BatchGenerator
        >>> from src.walks import DegreeBiasedNegativeSampler
        >>>
        >>> # Setup
        >>> pairs = [(0, 1), (1, 2), (2, 3), ...]
        >>> neg_sampler = DegreeBiasedNegativeSampler(edge_index, num_nodes)
        >>>
        >>> generator = BatchGenerator(
        ...     positive_pairs=pairs,
        ...     negative_sampler=neg_sampler,
        ...     batch_size=512,
        ...     num_negatives=5
        ... )
        >>>
        >>> # Iterate over batches
        >>> for batch in generator:
        ...     print(batch.targets.shape)  # [512]
        ...     print(batch.negatives.shape)  # [512, 5]
    """

    def __init__(
        self,
        positive_pairs: List[Tuple[int, int]],
        negative_sampler,
        batch_size: int = 512,
        num_negatives: int = 5,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        drop_last: bool = False
    ):
        """
        Initialize batch generator.

        Args:
            positive_pairs: List of (target, context) tuples
            negative_sampler: Sampler for negative nodes
            batch_size: Number of pairs per batch
            num_negatives: Negatives per positive pair
            device: Device for output tensors
            seed: Random seed for shuffling
            drop_last: Whether to drop incomplete last batch
        """
        self.positive_pairs = positive_pairs
        self.negative_sampler = negative_sampler
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.device = device or torch.device('cpu')
        self.seed = seed
        self.drop_last = drop_last

        # Pre-extract pairs for faster iteration
        self.targets = [p[0] for p in positive_pairs]
        self.contexts = [p[1] for p in positive_pairs]

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.positive_pairs)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        """
        Iterate over batches for one epoch.

        Shuffles pairs and generates batches with fresh negatives.
        """
        # Shuffle indices
        indices = list(range(len(self.positive_pairs)))
        random.shuffle(indices)

        # Generate batches
        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = start_idx + self.batch_size

            # Get batch indices
            batch_indices = indices[start_idx:end_idx]

            # Skip incomplete last batch if drop_last
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Extract targets and contexts
            batch_targets = torch.tensor(
                [self.targets[i] for i in batch_indices],
                dtype=torch.long,
                device=self.device
            )
            batch_contexts = torch.tensor(
                [self.contexts[i] for i in batch_indices],
                dtype=torch.long,
                device=self.device
            )

            # Sample negatives on-the-fly
            batch_negatives = self.negative_sampler.sample(
                len(batch_indices),
                self.num_negatives
            )
            # Ensure on correct device
            if batch_negatives.device != self.device:
                batch_negatives = batch_negatives.to(self.device)

            yield Batch(
                targets=batch_targets,
                contexts=batch_contexts,
                negatives=batch_negatives,
                size=len(batch_indices)
            )

    def get_epoch_batches(self) -> List[Batch]:
        """
        Get all batches for one epoch as a list.

        Useful when you need random access to batches.
        """
        return list(self)


class StreamingBatchGenerator:
    """
    Memory-efficient batch generator for very large pair sets.

    Instead of keeping all pairs in memory, this generates pairs
    on-the-fly from walks. Useful when the number of pairs would
    exceed available memory.
    """

    def __init__(
        self,
        walks: List[List[int]],
        context_window: int,
        negative_sampler,
        batch_size: int = 512,
        num_negatives: int = 5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize streaming generator.

        Args:
            walks: List of random walks
            context_window: Skip-gram context window size
            negative_sampler: Sampler for negatives
            batch_size: Pairs per batch
            num_negatives: Negatives per positive
            device: Output device
        """
        self.walks = walks
        self.context_window = context_window
        self.negative_sampler = negative_sampler
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.device = device or torch.device('cpu')

        # Estimate total pairs
        self._estimate_num_pairs()

    def _estimate_num_pairs(self):
        """Estimate number of pairs for progress tracking."""
        total = 0
        for walk in self.walks:
            walk_len = len(walk)
            # Each position generates ~2*context_window pairs
            pairs_per_pos = min(2 * self.context_window, walk_len - 1)
            total += walk_len * pairs_per_pos
        self.estimated_pairs = total

    def __iter__(self) -> Iterator[Batch]:
        """Generate batches by streaming through walks."""
        # Buffer to accumulate pairs until we have a batch
        target_buffer = []
        context_buffer = []

        # Shuffle walk order
        walk_indices = list(range(len(self.walks)))
        random.shuffle(walk_indices)

        for walk_idx in walk_indices:
            walk = self.walks[walk_idx]
            walk_len = len(walk)

            for i, target in enumerate(walk):
                # Context window
                start = max(0, i - self.context_window)
                end = min(walk_len, i + self.context_window + 1)

                for j in range(start, end):
                    if i != j:
                        target_buffer.append(target)
                        context_buffer.append(walk[j])

                        # Yield batch when buffer is full
                        if len(target_buffer) >= self.batch_size:
                            yield self._create_batch(
                                target_buffer[:self.batch_size],
                                context_buffer[:self.batch_size]
                            )
                            target_buffer = target_buffer[self.batch_size:]
                            context_buffer = context_buffer[self.batch_size:]

        # Yield remaining pairs
        if target_buffer:
            yield self._create_batch(target_buffer, context_buffer)

    def _create_batch(
        self,
        targets: List[int],
        contexts: List[int]
    ) -> Batch:
        """Create batch from lists."""
        batch_targets = torch.tensor(targets, dtype=torch.long, device=self.device)
        batch_contexts = torch.tensor(contexts, dtype=torch.long, device=self.device)
        batch_negatives = self.negative_sampler.sample(
            len(targets),
            self.num_negatives
        ).to(self.device)

        return Batch(
            targets=batch_targets,
            contexts=batch_contexts,
            negatives=batch_negatives,
            size=len(targets)
        )


def split_pairs_train_val(
    pairs: List[Tuple[int, int]],
    val_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Split positive pairs into train and validation sets.

    Args:
        pairs: All positive pairs
        val_fraction: Fraction for validation
        seed: Random seed

    Returns:
        (train_pairs, val_pairs)
    """
    random.seed(seed)
    pairs = list(pairs)  # Copy to avoid modifying original
    random.shuffle(pairs)

    split_idx = int(len(pairs) * (1 - val_fraction))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    return train_pairs, val_pairs
