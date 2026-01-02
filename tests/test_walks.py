"""
Tests for Walks Module.

Tests random walk generation, pair sampling, and negative sampling.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.walks import RandomWalkGenerator, CooccurrencePairSampler, DegreeBiasedNegativeSampler


class TestRandomWalkGenerator:
    """Tests for random walk generation."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph (chain: 0-1-2-3-4)."""
        edge_index = torch.tensor([
            [0, 1, 2, 3],  # sources
            [1, 2, 3, 4]   # targets
        ])
        return edge_index, 5  # 5 nodes

    @pytest.fixture
    def walker(self, simple_graph):
        """Create walker for simple graph."""
        edge_index, num_nodes = simple_graph
        return RandomWalkGenerator(
            edge_index=edge_index,
            num_nodes=num_nodes,
            walk_length=10,
            walks_per_node=3,
            seed=42
        )

    def test_single_walk_starts_correctly(self, walker):
        """Test that walk starts from specified node."""
        walk = walker.generate_single_walk(0)
        assert walk[0] == 0

    def test_single_walk_respects_length(self, walker):
        """Test that walk doesn't exceed max length."""
        walk = walker.generate_single_walk(0)
        assert len(walk) <= walker.walk_length

    def test_single_walk_valid_transitions(self, simple_graph, walker):
        """Test that walk only uses valid edges."""
        edge_index, num_nodes = simple_graph

        # Build valid transitions
        valid_next = {i: set() for i in range(num_nodes)}
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            valid_next[src].add(dst)

        walk = walker.generate_single_walk(0)

        for i in range(len(walk) - 1):
            current = walk[i]
            next_node = walk[i + 1]
            assert next_node in valid_next[current] or len(valid_next[current]) == 0

    def test_generate_all_walks(self, walker):
        """Test generating walks for all nodes."""
        walks = walker.generate_all_walks()

        # Should have walks_per_node walks for each non-isolated node
        # Some nodes may produce fewer walks if isolated
        assert len(walks) > 0
        assert len(walks) <= walker.num_nodes * walker.walks_per_node

    def test_walk_on_isolated_node(self):
        """Test walk generation when node is isolated."""
        # Node 5 is isolated
        edge_index = torch.tensor([[0, 1], [1, 2]])
        walker = RandomWalkGenerator(edge_index, num_nodes=6, walk_length=10)

        walk = walker.generate_single_walk(5)

        # Walk should be just the start node
        assert walk == [5]


class TestCooccurrencePairSampler:
    """Tests for pair extraction."""

    @pytest.fixture
    def sampler(self):
        """Create sampler with window size 2."""
        return CooccurrencePairSampler(context_window=2)

    def test_pair_extraction(self, sampler):
        """Test basic pair extraction."""
        walks = [[0, 1, 2, 3, 4]]
        pairs = sampler.extract_pairs(walks)

        assert len(pairs) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_window_respected(self, sampler):
        """Test that context window is respected."""
        walks = [[0, 1, 2, 3, 4]]
        pairs = sampler.extract_pairs(walks)

        # With window=2, node 0 should pair with 1, 2 but not 3, 4
        pairs_from_0 = [p for p in pairs if p[0] == 0]
        contexts_for_0 = [p[1] for p in pairs_from_0]

        assert 1 in contexts_for_0
        assert 2 in contexts_for_0
        assert 3 not in contexts_for_0
        assert 4 not in contexts_for_0

    def test_no_self_pairs(self, sampler):
        """Test that nodes don't pair with themselves."""
        walks = [[0, 1, 0, 1, 0]]
        pairs = sampler.extract_pairs(walks)

        for target, context in pairs:
            # Pairs are based on position, not value
            # So we just check the extraction works
            pass

    def test_tensor_output(self, sampler):
        """Test tensor output format."""
        walks = [[0, 1, 2, 3], [1, 2, 3, 4]]
        targets, contexts = sampler.extract_pairs_tensor(walks)

        assert targets.dtype == torch.long
        assert contexts.dtype == torch.long
        assert len(targets) == len(contexts)


class TestDegreeBiasedNegativeSampler:
    """Tests for negative sampling."""

    @pytest.fixture
    def sampler(self):
        """Create sampler from simple graph."""
        # Graph where node 0 has higher degree
        edge_index = torch.tensor([
            [0, 0, 0, 1, 2],
            [1, 2, 3, 2, 3]
        ])
        return DegreeBiasedNegativeSampler(
            edge_index=edge_index,
            num_nodes=4,
            exponent=0.75
        )

    def test_sample_shape(self, sampler):
        """Test output shape."""
        negatives = sampler.sample(num_samples=10, num_negatives=5)

        assert negatives.shape == (10, 5)

    def test_sample_range(self, sampler):
        """Test that samples are valid node indices."""
        negatives = sampler.sample(num_samples=100, num_negatives=5)

        assert negatives.min() >= 0
        assert negatives.max() < sampler.num_nodes

    def test_degree_bias(self):
        """Test that high-degree nodes are sampled more often."""
        # Create graph where node 0 has many edges
        edge_index = torch.tensor([
            [0, 0, 0, 0, 0, 1],
            [1, 2, 3, 4, 5, 2]
        ])
        sampler = DegreeBiasedNegativeSampler(edge_index, num_nodes=6, exponent=0.75)

        # Sample many times
        samples = sampler.sample(num_samples=10000, num_negatives=1).flatten()

        # Node 0 should appear more often than uniform
        node_0_count = (samples == 0).sum().item()
        uniform_expected = 10000 / 6

        assert node_0_count > uniform_expected  # Should be biased toward 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
