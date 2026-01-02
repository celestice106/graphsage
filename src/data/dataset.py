"""
PyTorch Dataset Module for GraphSAGE Training.

This module provides dataset wrappers and utilities for creating
PyTorch Geometric data objects from the extracted graph view and features.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

from .graph_loader import MockGraphStore, GraphLoader
from .view_extractor import GraphSAGEViewExtractor, GraphSAGEView
from .feature_extractor import MemoryFeatureExtractor


def create_data_object(
    view: GraphSAGEView,
    features: torch.Tensor,
    device: Optional[torch.device] = None
) -> Data:
    """
    Create a PyTorch Geometric Data object from view and features.

    The Data object is the standard input format for PyTorch Geometric models.
    It contains:
    - x: Node features [num_nodes, num_features]
    - edge_index: Edge indices [2, num_edges]
    - num_nodes: Number of nodes

    Args:
        view: Extracted GraphSAGE view
        features: Node features tensor
        device: Device to place tensors on (optional)

    Returns:
        PyTorch Geometric Data object
    """
    data = Data(
        x=features,
        edge_index=view.edge_index,
        num_nodes=view.num_nodes
    )

    if device is not None:
        data = data.to(device)

    return data


class GraphSAGEDataset:
    """
    Complete dataset for GraphSAGE training.

    This class encapsulates all data needed for training:
    - Graph structure (nodes, edges)
    - Node features
    - Mappings between original IDs and indices
    - Full graph reference for feature computation

    It provides a clean interface for the training pipeline and handles
    all data preparation steps.

    Example:
        >>> from src.data import GraphSAGEDataset
        >>>
        >>> # Create dataset from mock graph
        >>> dataset = GraphSAGEDataset.from_mock(num_memories=500)
        >>>
        >>> # Get PyTorch Geometric data object
        >>> data = dataset.get_data()
        >>> print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
        >>>
        >>> # Move to GPU
        >>> data = dataset.get_data(device=torch.device('cuda'))
    """

    def __init__(
        self,
        full_graph: MockGraphStore,
        view: GraphSAGEView,
        features: torch.Tensor
    ):
        """
        Initialize dataset.

        Args:
            full_graph: Full Memory R1 graph
            view: Extracted GraphSAGE view
            features: Computed node features
        """
        self.full_graph = full_graph
        self.view = view
        self.features = features

        # Create PyG data object (on CPU by default)
        self._data = create_data_object(view, features)

    @classmethod
    def from_mock(
        cls,
        num_memories: int = 200,
        num_entities: int = 50,
        seed: int = 42,
        undirected: bool = True,
        include_entity_features: bool = True,
        **kwargs
    ) -> 'GraphSAGEDataset':
        """
        Create dataset from a mock graph.

        This is the recommended way to create a dataset for testing
        and development.

        Args:
            num_memories: Number of memory nodes
            num_entities: Number of entity nodes
            seed: Random seed for reproducibility
            undirected: Whether to treat edges as undirected
            include_entity_features: Whether to include entity-derived features
            **kwargs: Additional arguments for graph generation

        Returns:
            GraphSAGEDataset ready for training
        """
        # Step 1: Create mock graph
        full_graph = GraphLoader.create_mock(
            num_memories=num_memories,
            num_entities=num_entities,
            seed=seed,
            **kwargs
        )

        # Step 2: Extract view
        extractor = GraphSAGEViewExtractor(full_graph)
        if undirected:
            view = extractor.extract_with_undirected_edges()
        else:
            view = extractor.extract()

        # Step 3: Compute features
        feature_extractor = MemoryFeatureExtractor(
            full_graph=full_graph,
            node_mapping=view.node_mapping,
            include_entity_features=include_entity_features
        )
        features = feature_extractor.extract(view.edge_index)

        return cls(full_graph, view, features)

    @classmethod
    def from_file(
        cls,
        graph_path: str,
        undirected: bool = True,
        include_entity_features: bool = True
    ) -> 'GraphSAGEDataset':
        """
        Create dataset from a saved graph file.

        Args:
            graph_path: Path to JSON graph file
            undirected: Whether to treat edges as undirected
            include_entity_features: Whether to include entity-derived features

        Returns:
            GraphSAGEDataset
        """
        # Load graph
        full_graph = GraphLoader.from_file(graph_path)

        # Extract view
        extractor = GraphSAGEViewExtractor(full_graph)
        if undirected:
            view = extractor.extract_with_undirected_edges()
        else:
            view = extractor.extract()

        # Compute features
        feature_extractor = MemoryFeatureExtractor(
            full_graph=full_graph,
            node_mapping=view.node_mapping,
            include_entity_features=include_entity_features
        )
        features = feature_extractor.extract(view.edge_index)

        return cls(full_graph, view, features)

    @classmethod
    def from_memory_r1(
        cls,
        memory_bank: Any,
        undirected: bool = True,
        include_entity_features: bool = True
    ) -> 'GraphSAGEDataset':
        """
        Create dataset from Memory R1 bank.

        Args:
            memory_bank: Memory R1 bank instance
            undirected: Whether to treat edges as undirected
            include_entity_features: Whether to include entity-derived features

        Returns:
            GraphSAGEDataset
        """
        # Load from Memory R1
        full_graph = GraphLoader.from_memory_r1(memory_bank)

        # Extract view
        extractor = GraphSAGEViewExtractor(full_graph)
        if undirected:
            view = extractor.extract_with_undirected_edges()
        else:
            view = extractor.extract()

        # Compute features
        feature_extractor = MemoryFeatureExtractor(
            full_graph=full_graph,
            node_mapping=view.node_mapping,
            include_entity_features=include_entity_features
        )
        features = feature_extractor.extract(view.edge_index)

        return cls(full_graph, view, features)

    def get_data(self, device: Optional[torch.device] = None) -> Data:
        """
        Get PyTorch Geometric Data object.

        Args:
            device: Device to move data to (optional)

        Returns:
            Data object with features and edges
        """
        if device is not None:
            return self._data.to(device)
        return self._data

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.view.num_nodes

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.view.num_edges

    @property
    def num_features(self) -> int:
        """Number of features per node."""
        return self.features.shape[1]

    @property
    def node_mapping(self) -> Dict[str, int]:
        """Mapping from original memory ID to index."""
        return self.view.node_mapping

    @property
    def reverse_mapping(self) -> Dict[int, str]:
        """Mapping from index to original memory ID."""
        return self.view.reverse_mapping

    def get_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics.

        Returns:
            Dictionary with graph and feature statistics
        """
        # Graph statistics
        edge_index = self.view.edge_index

        if self.num_edges > 0:
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

            out_degrees = torch.bincount(src_nodes, minlength=self.num_nodes)
            in_degrees = torch.bincount(dst_nodes, minlength=self.num_nodes)
            total_degrees = out_degrees + in_degrees
        else:
            total_degrees = torch.zeros(self.num_nodes)

        # Feature statistics
        feature_stats = {}
        feature_names = [
            'caused_by_degree', 'next_event_degree', 'num_entities_mentioned',
            'shared_entity_neighbors', 'is_cause', 'is_effect', 'has_successor'
        ]
        for i, name in enumerate(feature_names):
            col = self.features[:, i]
            feature_stats[name] = {
                'mean': col.mean().item(),
                'std': col.std().item(),
                'min': col.min().item(),
                'max': col.max().item()
            }

        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'num_features': self.num_features,
            'avg_degree': total_degrees.float().mean().item(),
            'max_degree': total_degrees.max().item() if self.num_nodes > 0 else 0,
            'isolated_nodes': (total_degrees == 0).sum().item(),
            'density': self.num_edges / (self.num_nodes * (self.num_nodes - 1))
                       if self.num_nodes > 1 else 0,
            'feature_stats': feature_stats
        }

    def save(self, path: str) -> None:
        """
        Save dataset to disk.

        Saves:
        - Graph structure (JSON)
        - Features (PyTorch tensor)
        - Mappings (JSON)

        Args:
            path: Directory path to save to
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save full graph
        self.full_graph.save(str(save_dir / 'full_graph.json'))

        # Save features
        torch.save(self.features, save_dir / 'features.pt')

        # Save edge index
        torch.save(self.view.edge_index, save_dir / 'edge_index.pt')

        # Save mappings
        with open(save_dir / 'mappings.json', 'w') as f:
            json.dump({
                'node_mapping': self.view.node_mapping,
                'memory_ids': self.view.memory_ids
            }, f)

        # Save statistics
        with open(save_dir / 'statistics.json', 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'GraphSAGEDataset':
        """
        Load dataset from disk.

        Args:
            path: Directory path to load from

        Returns:
            GraphSAGEDataset
        """
        load_dir = Path(path)

        # Load full graph
        full_graph = GraphLoader.from_file(str(load_dir / 'full_graph.json'))

        # Load features
        features = torch.load(load_dir / 'features.pt')

        # Load edge index
        edge_index = torch.load(load_dir / 'edge_index.pt')

        # Load mappings
        with open(load_dir / 'mappings.json', 'r') as f:
            mappings = json.load(f)

        # Reconstruct view
        node_mapping = mappings['node_mapping']
        memory_ids = mappings['memory_ids']
        reverse_mapping = {int(idx): mid for mid, idx in node_mapping.items()}

        view = GraphSAGEView(
            memory_ids=memory_ids,
            edge_index=edge_index,
            node_mapping=node_mapping,
            reverse_mapping=reverse_mapping,
            num_nodes=len(memory_ids),
            num_edges=edge_index.shape[1]
        )

        return cls(full_graph, view, features)
