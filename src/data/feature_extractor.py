"""
Memory Feature Extractor Module.

This module computes 7-dimensional features for each memory node. The features
capture both structural information from the memory-only subgraph and semantic
information derived from entity mentions.

Feature Design (7 dimensions):
    0. caused_by_degree: Number of causal connections (in + out)
    1. next_event_degree: Number of temporal connections (in + out)
    2. num_entities_mentioned: How many entities this memory references
    3. shared_entity_neighbors: Memories sharing at least one entity
    4. is_cause: Binary - has outgoing caused_by edge
    5. is_effect: Binary - has incoming caused_by edge
    6. has_successor: Binary - has outgoing next_event edge

Normalization:
    - Continuous features (0-3) use log normalization: log1p(x) / max(log1p(x))
    - Binary features (4-6) remain as 0.0 or 1.0
"""

import torch
from typing import Dict, List, Optional, Set
from collections import defaultdict

from .graph_loader import MockGraphStore, EdgeType
from .view_extractor import GraphSAGEView


class MemoryFeatureExtractor:
    """
    Compute 7-dimensional features for memory nodes.

    This class computes features that capture:
    1. Structural role in the graph (degree, connectivity patterns)
    2. Entity-derived information (how memories relate through shared entities)

    Features are designed to be informative for the skip-gram objective,
    helping the model learn meaningful structural embeddings.

    Example:
        >>> from src.data import MockGraphStore, GraphSAGEViewExtractor
        >>> from src.data import MemoryFeatureExtractor
        >>>
        >>> # Setup
        >>> store = MockGraphStore()
        >>> store.generate_synthetic_graph(num_memories=500)
        >>> extractor = GraphSAGEViewExtractor(store)
        >>> view = extractor.extract()
        >>>
        >>> # Extract features
        >>> feature_extractor = MemoryFeatureExtractor(store, view.node_mapping)
        >>> features = feature_extractor.extract(view.edge_index)
        >>> print(f"Features shape: {features.shape}")  # [500, 7]
    """

    def __init__(
        self,
        full_graph: MockGraphStore,
        node_mapping: Dict[str, int],
        include_entity_features: bool = True
    ):
        """
        Initialize feature extractor.

        Args:
            full_graph: Full Memory R1 graph (needed for entity information)
            node_mapping: Mapping from original memory ID to index
            include_entity_features: Whether to compute entity-derived features
        """
        self.full_graph = full_graph
        self.node_mapping = node_mapping
        self.reverse_mapping = {idx: mid for mid, idx in node_mapping.items()}
        self.num_nodes = len(node_mapping)
        self.include_entity_features = include_entity_features

        # Pre-compute edge type specific degrees
        self._precompute_edge_info()

    def _precompute_edge_info(self) -> None:
        """
        Pre-compute edge type specific information.

        This caches degree counts and edge direction info for each edge type,
        avoiding repeated iteration over edges during feature extraction.
        """
        # Degree counters per edge type
        self.caused_by_out: Dict[str, int] = defaultdict(int)
        self.caused_by_in: Dict[str, int] = defaultdict(int)
        self.next_event_out: Dict[str, int] = defaultdict(int)
        self.next_event_in: Dict[str, int] = defaultdict(int)

        # Get caused_by edges
        for src, dst, _ in self.full_graph.get_edges_by_type(["caused_by"]):
            if src in self.node_mapping and dst in self.node_mapping:
                self.caused_by_out[src] += 1
                self.caused_by_in[dst] += 1

        # Get next_event edges
        for src, dst, _ in self.full_graph.get_edges_by_type(["next_event"]):
            if src in self.node_mapping and dst in self.node_mapping:
                self.next_event_out[src] += 1
                self.next_event_in[dst] += 1

    def extract(
        self,
        edge_index: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute 7-dimensional features for each memory node.

        Args:
            edge_index: Edge indices (not directly used, but kept for interface)
            normalize: Whether to apply log normalization to continuous features

        Returns:
            features: Tensor of shape [num_nodes, 7]
        """
        # Initialize feature tensor
        # Using float32 for compatibility (will be cast to GPU dtype as needed)
        features = torch.zeros(self.num_nodes, 7, dtype=torch.float32)

        for original_id, idx in self.node_mapping.items():
            # === Feature 0: caused_by degree (in + out) ===
            # Higher values indicate nodes central to causal reasoning
            caused_by_deg = (
                self.caused_by_out.get(original_id, 0) +
                self.caused_by_in.get(original_id, 0)
            )
            features[idx, 0] = float(caused_by_deg)

            # === Feature 1: next_event degree (in + out) ===
            # Higher values indicate nodes in longer temporal chains
            next_event_deg = (
                self.next_event_out.get(original_id, 0) +
                self.next_event_in.get(original_id, 0)
            )
            features[idx, 1] = float(next_event_deg)

            # === Features 2-3: Entity-derived features ===
            if self.include_entity_features:
                # Feature 2: Number of entities mentioned
                # Memories referencing many entities are often important facts
                entities = self.full_graph.get_mentioned_entities(original_id)
                features[idx, 2] = float(len(entities))

                # Feature 3: Shared entity neighbors
                # Count of other memories sharing at least one entity
                shared_neighbors = self._count_shared_entity_neighbors(original_id)
                features[idx, 3] = float(shared_neighbors)

            # === Features 4-6: Binary structural role indicators ===

            # Feature 4: is_cause (has outgoing caused_by)
            # True if this memory causes other events
            features[idx, 4] = 1.0 if self.caused_by_out.get(original_id, 0) > 0 else 0.0

            # Feature 5: is_effect (has incoming caused_by)
            # True if this memory is caused by other events
            features[idx, 5] = 1.0 if self.caused_by_in.get(original_id, 0) > 0 else 0.0

            # Feature 6: has_successor (has outgoing next_event)
            # True if there's a recorded event after this one
            features[idx, 6] = 1.0 if self.next_event_out.get(original_id, 0) > 0 else 0.0

        # Apply normalization to continuous features (columns 0-3)
        if normalize:
            features[:, :4] = self._log_normalize(features[:, :4])

        return features

    def _count_shared_entity_neighbors(self, memory_id: str) -> int:
        """
        Count memories that share at least one entity with given memory.

        This feature captures semantic relatedness through shared entities.
        Two memories mentioning the same entity are likely related.

        Args:
            memory_id: ID of the memory node

        Returns:
            Number of other memories sharing at least one entity
        """
        # Get entities mentioned by this memory
        my_entities = set(self.full_graph.get_mentioned_entities(memory_id))

        if not my_entities:
            return 0

        # Find all memories sharing any entity
        neighbors: Set[str] = set()
        for entity_id in my_entities:
            mentioning = self.full_graph.get_memories_mentioning(entity_id)
            # Only count memories that are in our view
            for mid in mentioning:
                if mid in self.node_mapping:
                    neighbors.add(mid)

        # Remove self
        neighbors.discard(memory_id)

        return len(neighbors)

    def _log_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply log normalization to prevent large value dominance.

        Uses log1p (log(1 + x)) to handle zeros gracefully, then
        scales to [0, 1] range per column.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor with same shape
        """
        # log1p handles zeros: log1p(0) = 0
        log_x = torch.log1p(x)

        # Per-column normalization to [0, 1]
        max_vals = log_x.max(dim=0).values
        # Avoid division by zero
        max_vals = max_vals.clamp(min=1e-8)

        return log_x / max_vals

    def get_feature_names(self) -> List[str]:
        """Get names of each feature dimension."""
        return [
            'caused_by_degree',
            'next_event_degree',
            'num_entities_mentioned',
            'shared_entity_neighbors',
            'is_cause',
            'is_effect',
            'has_successor'
        ]

    def get_statistics(self) -> Dict:
        """
        Get statistics about extracted features.

        Returns:
            Dictionary with feature statistics
        """
        features = self.extract(normalize=False)

        stats = {}
        feature_names = self.get_feature_names()

        for i, name in enumerate(feature_names):
            col = features[:, i]
            stats[name] = {
                'mean': col.mean().item(),
                'std': col.std().item(),
                'min': col.min().item(),
                'max': col.max().item(),
                'nonzero_fraction': (col > 0).float().mean().item()
            }

        return stats


def compute_features_for_view(
    full_graph: MockGraphStore,
    view: GraphSAGEView,
    include_entity_features: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    Convenience function to compute features for a GraphSAGE view.

    Args:
        full_graph: Full Memory R1 graph
        view: Extracted GraphSAGE view
        include_entity_features: Whether to include entity-derived features
        normalize: Whether to apply normalization

    Returns:
        Features tensor of shape [num_nodes, 7]
    """
    extractor = MemoryFeatureExtractor(
        full_graph=full_graph,
        node_mapping=view.node_mapping,
        include_entity_features=include_entity_features
    )
    return extractor.extract(view.edge_index, normalize=normalize)
