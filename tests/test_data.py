"""
Tests for Data Module.

Tests data loading, view extraction, and feature computation.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import GraphLoader, GraphSAGEDataset
from src.data.graph_loader import MockGraphStore, MemoryNode, EntityNode, Edge, EdgeType
from src.data.view_extractor import GraphSAGEViewExtractor
from src.data.feature_extractor import MemoryFeatureExtractor


class TestMockGraphStore:
    """Tests for MockGraphStore."""

    def test_empty_store(self):
        """Test empty store initialization."""
        store = MockGraphStore()
        assert len(store.memory_nodes) == 0
        assert len(store.entity_nodes) == 0

    def test_add_memory_node(self):
        """Test adding memory nodes."""
        store = MockGraphStore()
        node = MemoryNode(id="mem_001", content="Test memory")
        store.add_memory_node(node)

        assert len(store.memory_nodes) == 1
        assert "mem_001" in store.memory_nodes

    def test_add_edge(self):
        """Test adding edges."""
        store = MockGraphStore()

        # Add nodes first
        store.add_memory_node(MemoryNode(id="mem_001"))
        store.add_memory_node(MemoryNode(id="mem_002"))

        # Add edge
        edge = Edge(source="mem_001", target="mem_002", edge_type=EdgeType.CAUSED_BY)
        store.add_edge(edge)

        assert len(store.edges_by_type[EdgeType.CAUSED_BY]) == 1

    def test_generate_synthetic_graph(self):
        """Test synthetic graph generation."""
        store = MockGraphStore()
        store.generate_synthetic_graph(
            num_memories=100,
            num_entities=20,
            seed=42
        )

        assert len(store.memory_nodes) == 100
        assert len(store.entity_nodes) == 20
        assert len(store.edges_by_type[EdgeType.CAUSED_BY]) > 0
        assert len(store.edges_by_type[EdgeType.MENTION]) > 0


class TestGraphSAGEViewExtractor:
    """Tests for view extraction."""

    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph for testing."""
        store = MockGraphStore()
        store.generate_synthetic_graph(num_memories=50, num_entities=10, seed=42)
        return store

    def test_extract_view(self, mock_graph):
        """Test basic view extraction."""
        extractor = GraphSAGEViewExtractor(mock_graph)
        view = extractor.extract()

        assert view.num_nodes == 50
        assert len(view.memory_ids) == 50
        assert len(view.node_mapping) == 50
        assert view.edge_index.shape[0] == 2

    def test_extract_undirected(self, mock_graph):
        """Test undirected view extraction."""
        extractor = GraphSAGEViewExtractor(mock_graph)

        directed = extractor.extract()
        undirected = extractor.extract_with_undirected_edges()

        # Undirected should have more edges (reverse edges added)
        assert undirected.num_edges >= directed.num_edges

    def test_node_mapping_consistency(self, mock_graph):
        """Test that node mapping is consistent."""
        extractor = GraphSAGEViewExtractor(mock_graph)
        view = extractor.extract()

        for memory_id in view.memory_ids:
            idx = view.node_mapping[memory_id]
            assert view.reverse_mapping[idx] == memory_id


class TestFeatureExtractor:
    """Tests for feature extraction."""

    @pytest.fixture
    def setup_data(self):
        """Create graph and view for testing."""
        store = MockGraphStore()
        store.generate_synthetic_graph(num_memories=50, num_entities=10, seed=42)

        extractor = GraphSAGEViewExtractor(store)
        view = extractor.extract_with_undirected_edges()

        return store, view

    def test_feature_dimensions(self, setup_data):
        """Test feature dimensions."""
        store, view = setup_data

        feature_extractor = MemoryFeatureExtractor(
            full_graph=store,
            node_mapping=view.node_mapping
        )
        features = feature_extractor.extract(view.edge_index)

        assert features.shape == (view.num_nodes, 7)

    def test_feature_normalization(self, setup_data):
        """Test that continuous features are normalized."""
        store, view = setup_data

        feature_extractor = MemoryFeatureExtractor(
            full_graph=store,
            node_mapping=view.node_mapping
        )
        features = feature_extractor.extract(view.edge_index, normalize=True)

        # Continuous features (0-3) should be in [0, 1]
        assert features[:, :4].min() >= 0
        assert features[:, :4].max() <= 1.0 + 1e-6

    def test_binary_features(self, setup_data):
        """Test that binary features are 0 or 1."""
        store, view = setup_data

        feature_extractor = MemoryFeatureExtractor(
            full_graph=store,
            node_mapping=view.node_mapping
        )
        features = feature_extractor.extract(view.edge_index)

        # Binary features (4-6) should be 0 or 1
        binary = features[:, 4:7]
        assert ((binary == 0) | (binary == 1)).all()


class TestGraphSAGEDataset:
    """Tests for dataset wrapper."""

    def test_from_mock(self):
        """Test dataset creation from mock."""
        dataset = GraphSAGEDataset.from_mock(
            num_memories=100,
            num_entities=20,
            seed=42
        )

        assert dataset.num_nodes == 100
        assert dataset.num_features == 7
        assert dataset.num_edges > 0

    def test_get_data(self):
        """Test PyG data object creation."""
        dataset = GraphSAGEDataset.from_mock(num_memories=50)
        data = dataset.get_data()

        assert data.x.shape == (50, 7)
        assert data.edge_index.shape[0] == 2
        assert data.num_nodes == 50

    def test_device_placement(self):
        """Test data device placement."""
        dataset = GraphSAGEDataset.from_mock(num_memories=50)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            data = dataset.get_data(device=device)
            assert data.x.device.type == 'cuda'
            assert data.edge_index.device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
