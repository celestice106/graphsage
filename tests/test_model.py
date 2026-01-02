"""
Tests for Model Module.

Tests GraphSAGE model and loss functions.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import ProductionGraphSAGE, SkipGramLoss
from src.model.loss import MarginRankingLoss, InfoNCELoss


class TestProductionGraphSAGE:
    """Tests for GraphSAGE model."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return ProductionGraphSAGE(
            in_channels=7,
            hidden_channels=64,
            out_channels=64,
            num_layers=2,
            dropout=0.3
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        num_nodes = 100
        num_edges = 500

        features = torch.randn(num_nodes, 7)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        return features, edge_index

    def test_forward_shape(self, model, sample_data):
        """Test forward pass output shape."""
        features, edge_index = sample_data
        model.eval()

        with torch.no_grad():
            output = model(features, edge_index)

        assert output.shape == (100, 64)

    def test_output_normalized(self, model, sample_data):
        """Test that output embeddings are L2 normalized."""
        features, edge_index = sample_data
        model.eval()

        with torch.no_grad():
            output = model(features, edge_index)

        norms = output.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow through the model."""
        features, edge_index = sample_data
        model.train()

        output = model(features, edge_index)
        loss = output.sum()
        loss.backward()

        # Check that parameters have gradients
        for param in model.parameters():
            assert param.grad is not None

    def test_parameter_count(self, model):
        """Test parameter counting."""
        count = model.count_parameters()
        assert count > 0
        # Rough estimate for 2-layer model: 7*64 + 64*64 + biases
        assert count < 100000  # Sanity check

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self, model, sample_data):
        """Test forward pass on GPU."""
        features, edge_index = sample_data

        model = model.cuda()
        features = features.cuda()
        edge_index = edge_index.cuda()

        with torch.no_grad():
            output = model(features, edge_index)

        assert output.device.type == 'cuda'
        assert output.shape == (100, 64)


class TestSkipGramLoss:
    """Tests for skip-gram loss."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function."""
        return SkipGramLoss()

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        num_nodes = 100
        batch_size = 32
        num_negatives = 5

        embeddings = torch.randn(num_nodes, 64)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        targets = torch.randint(0, num_nodes, (batch_size,))
        contexts = torch.randint(0, num_nodes, (batch_size,))
        negatives = torch.randint(0, num_nodes, (batch_size, num_negatives))

        return embeddings, targets, contexts, negatives

    def test_loss_positive(self, loss_fn, sample_batch):
        """Test that loss is positive."""
        embeddings, targets, contexts, negatives = sample_batch

        loss = loss_fn(embeddings, targets, contexts, negatives)

        assert loss.item() > 0

    def test_loss_gradient(self, loss_fn, sample_batch):
        """Test that loss has gradient."""
        embeddings, targets, contexts, negatives = sample_batch
        embeddings.requires_grad = True

        loss = loss_fn(embeddings, targets, contexts, negatives)
        loss.backward()

        assert embeddings.grad is not None

    def test_loss_with_details(self, loss_fn, sample_batch):
        """Test loss computation with details."""
        embeddings, targets, contexts, negatives = sample_batch

        loss, details = loss_fn.forward_with_details(
            embeddings, targets, contexts, negatives
        )

        assert 'total_loss' in details
        assert 'pos_loss' in details
        assert 'neg_loss' in details
        assert 'pos_prob' in details
        assert 'neg_prob' in details


class TestAlternativeLosses:
    """Tests for alternative loss functions."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        embeddings = torch.randn(50, 64)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        targets = torch.randint(0, 50, (16,))
        contexts = torch.randint(0, 50, (16,))
        negatives = torch.randint(0, 50, (16, 5))

        return embeddings, targets, contexts, negatives

    def test_margin_loss(self, sample_batch):
        """Test margin ranking loss."""
        loss_fn = MarginRankingLoss(margin=1.0)
        embeddings, targets, contexts, negatives = sample_batch

        loss = loss_fn(embeddings, targets, contexts, negatives)

        assert loss.item() >= 0

    def test_infonce_loss(self, sample_batch):
        """Test InfoNCE loss."""
        loss_fn = InfoNCELoss(temperature=0.07)
        embeddings, targets, contexts, negatives = sample_batch

        loss = loss_fn(embeddings, targets, contexts, negatives)

        assert loss.item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
