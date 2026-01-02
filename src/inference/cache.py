"""
Embedding Cache Module.

This module implements caching for computed embeddings with support for:
- Full cache invalidation (when graph structure changes significantly)
- Partial invalidation (when specific nodes change)
- Lazy recomputation on access

The cache is designed for the Memory R1 use case where the graph evolves
during RL training, requiring efficient embedding updates.
"""

import torch
from typing import Set, Optional, Dict, List
from dataclasses import dataclass, field
import time


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    embeddings: torch.Tensor
    timestamp: float
    version: int
    num_nodes: int


class EmbeddingCache:
    """
    Cache computed embeddings with dirty tracking.

    This cache tracks which nodes have been modified since the last
    embedding computation. When embeddings are requested:
    - If cache is valid (no dirty nodes), return cached embeddings
    - If cache is invalid, trigger recomputation

    The cache does NOT do incremental updates - any change invalidates
    the entire cache because GraphSAGE embeddings depend on neighbors.

    Example:
        >>> cache = EmbeddingCache()
        >>>
        >>> # Store embeddings
        >>> cache.update(embeddings)
        >>>
        >>> # Check validity
        >>> if cache.is_valid():
        ...     emb = cache.get_all()
        >>>
        >>> # Invalidate on graph change
        >>> cache.invalidate()  # Full invalidation
        >>> cache.invalidate([node_id])  # Partial (still triggers full recompute)
    """

    def __init__(self, max_age_seconds: Optional[float] = None):
        """
        Initialize embedding cache.

        Args:
            max_age_seconds: Optional maximum age before auto-invalidation
        """
        self.embeddings: Optional[torch.Tensor] = None
        self.valid: bool = False
        self.dirty_nodes: Set[int] = set()
        self.max_age_seconds = max_age_seconds

        # Metadata
        self.version: int = 0
        self.last_update_time: float = 0
        self.num_nodes: int = 0

        # Statistics
        self.hits: int = 0
        self.misses: int = 0
        self.invalidations: int = 0

    def is_valid(self) -> bool:
        """
        Check if cache is valid.

        Returns False if:
        - No embeddings stored
        - Explicit invalidation called
        - Any dirty nodes exist
        - Cache has expired (if max_age set)
        """
        if not self.valid or self.embeddings is None:
            return False

        if len(self.dirty_nodes) > 0:
            return False

        if self.max_age_seconds is not None:
            age = time.time() - self.last_update_time
            if age > self.max_age_seconds:
                return False

        return True

    def get_all(self) -> Optional[torch.Tensor]:
        """
        Get cached embeddings if valid.

        Returns:
            Cached embeddings or None if invalid
        """
        if self.is_valid():
            self.hits += 1
            return self.embeddings
        else:
            self.misses += 1
            return None

    def get_single(self, node_idx: int) -> Optional[torch.Tensor]:
        """
        Get embedding for single node if cache valid.

        Args:
            node_idx: Node index

        Returns:
            Single embedding [dim] or None if invalid
        """
        embeddings = self.get_all()
        if embeddings is not None and node_idx < embeddings.shape[0]:
            return embeddings[node_idx]
        return None

    def update(self, embeddings: torch.Tensor):
        """
        Update cache with new embeddings.

        Args:
            embeddings: New embeddings tensor [num_nodes, dim]
        """
        self.embeddings = embeddings
        self.valid = True
        self.dirty_nodes.clear()
        self.version += 1
        self.last_update_time = time.time()
        self.num_nodes = embeddings.shape[0]

    def invalidate(self, node_ids: Optional[List[int]] = None):
        """
        Invalidate cache.

        Args:
            node_ids: Specific nodes to mark dirty (None = full invalidation)
        """
        self.invalidations += 1

        if node_ids is None:
            # Full invalidation
            self.valid = False
            self.embeddings = None
            self.dirty_nodes.clear()
        else:
            # Mark specific nodes as dirty
            # Note: Due to GNN message passing, changing one node
            # affects its neighbors' embeddings too. For simplicity,
            # any dirty node triggers full recomputation.
            self.dirty_nodes.update(node_ids)

    def clear(self):
        """Clear cache completely."""
        self.embeddings = None
        self.valid = False
        self.dirty_nodes.clear()
        self.version = 0
        self.last_update_time = 0
        self.num_nodes = 0

    def get_statistics(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'valid': self.valid,
            'num_nodes': self.num_nodes,
            'version': self.version,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'invalidations': self.invalidations,
            'dirty_nodes': len(self.dirty_nodes),
            'age_seconds': time.time() - self.last_update_time
                          if self.last_update_time > 0 else None
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"EmbeddingCache(valid={stats['valid']}, "
                f"nodes={stats['num_nodes']}, "
                f"hit_rate={stats['hit_rate']:.2%})")


class LRUEmbeddingCache:
    """
    LRU cache for node embeddings.

    Alternative cache implementation that keeps embeddings for
    recently accessed nodes. Useful when only a subset of nodes
    are frequently accessed.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of nodes to cache
        """
        self.max_size = max_size
        self.cache: Dict[int, torch.Tensor] = {}
        self.access_order: List[int] = []

    def get(self, node_idx: int) -> Optional[torch.Tensor]:
        """Get embedding for node."""
        if node_idx in self.cache:
            # Move to end of access order (most recent)
            self.access_order.remove(node_idx)
            self.access_order.append(node_idx)
            return self.cache[node_idx]
        return None

    def put(self, node_idx: int, embedding: torch.Tensor):
        """Store embedding for node."""
        if node_idx not in self.cache:
            # Evict oldest if at capacity
            while len(self.cache) >= self.max_size:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]

            self.access_order.append(node_idx)

        self.cache[node_idx] = embedding

    def put_batch(self, node_indices: torch.Tensor, embeddings: torch.Tensor):
        """Store embeddings for batch of nodes."""
        for i, node_idx in enumerate(node_indices.tolist()):
            self.put(node_idx, embeddings[i])

    def invalidate(self, node_ids: Optional[List[int]] = None):
        """Invalidate specific nodes or all."""
        if node_ids is None:
            self.cache.clear()
            self.access_order.clear()
        else:
            for node_id in node_ids:
                if node_id in self.cache:
                    del self.cache[node_id]
                    self.access_order.remove(node_id)

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return f"LRUEmbeddingCache(size={len(self)}/{self.max_size})"
