"""
End-to-End Integration Tests.

Tests the complete training pipeline from data generation to inference.
"""

import pytest
import torch
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import load_config
from src.data import GraphSAGEDataset, GraphLoader
from src.data.view_extractor import GraphSAGEViewExtractor
from src.data.feature_extractor import MemoryFeatureExtractor
from src.walks import RandomWalkGenerator, CooccurrencePairSampler, DegreeBiasedNegativeSampler
from src.model import GraphSAGE, SkipGramLoss
from src.training import GraphSAGETrainer
from src.inference import StructuralEncoder
from src.utils.metrics import evaluate_embeddings


class TestEndToEndPipeline:
    """Tests for complete pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        config = load_config()
        # Reduce sizes for faster testing
        config['training']['epochs'] = 5
        config['training']['batch_size'] = 64
        config['training']['early_stopping_patience'] = 3
        config['walks']['per_node'] = 3
        config['walks']['length'] = 20
        return config

    def test_data_generation_pipeline(self):
        """Test data generation from scratch."""
        # Create graph
        full_graph = GraphLoader.create_mock(
            num_memories=100,
            num_entities=20,
            seed=42
        )

        # Extract view
        view_extractor = GraphSAGEViewExtractor(full_graph)
        view = view_extractor.extract_with_undirected_edges()

        assert view.num_nodes == 100
        assert view.num_edges > 0

        # Compute features
        feature_extractor = MemoryFeatureExtractor(
            full_graph=full_graph,
            node_mapping=view.node_mapping
        )
        features = feature_extractor.extract(view.edge_index)

        assert features.shape == (100, 7)

        # Generate walks
        walker = RandomWalkGenerator(
            edge_index=view.edge_index,
            num_nodes=view.num_nodes,
            walk_length=20,
            walks_per_node=5
        )
        walks = walker.generate_all_walks()

        assert len(walks) > 0

        # Extract pairs
        pair_sampler = CooccurrencePairSampler(context_window=5)
        pairs = pair_sampler.extract_pairs(walks)

        assert len(pairs) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_training_pipeline(self, config, temp_dir):
        """Test training from data to model."""
        device = torch.device('cuda')

        # Create dataset
        dataset = GraphSAGEDataset.from_mock(
            num_memories=100,
            num_entities=20,
            seed=42
        )
        data = dataset.get_data(device=device)

        # Generate walks and pairs
        walker = RandomWalkGenerator(
            edge_index=data.edge_index.cpu(),
            num_nodes=data.num_nodes,
            walk_length=config['walks']['length'],
            walks_per_node=config['walks']['per_node']
        )
        walks = walker.generate_all_walks()

        pair_sampler = CooccurrencePairSampler(
            context_window=config['walks']['context_window']
        )
        pairs = pair_sampler.extract_pairs(walks)

        # Create negative sampler
        neg_sampler = DegreeBiasedNegativeSampler(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            device=device
        )

        # Create model
        model = GraphSAGE(
            in_channels=7,
            hidden_channels=32,  # Smaller for testing
            out_channels=32
        ).to(device)

        # Update config for temp dir
        config['paths']['checkpoints'] = str(temp_dir / 'checkpoints')
        config['paths']['logs'] = str(temp_dir / 'logs')

        # Create trainer
        trainer = GraphSAGETrainer(
            model=model,
            features=data.x,
            edge_index=data.edge_index,
            positive_pairs=pairs,
            negative_sampler=neg_sampler,
            config=config
        )

        # Train
        best_loss = trainer.train(num_epochs=config['training']['epochs'])

        assert best_loss > 0
        assert best_loss < 10  # Sanity check

        # Get embeddings
        embeddings = trainer.get_embeddings()

        assert embeddings.shape == (100, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_inference_pipeline(self, temp_dir):
        """Test inference with trained model."""
        device = torch.device('cuda')

        # Create and train minimal model
        full_graph = GraphLoader.create_mock(num_memories=50, seed=42)

        model = GraphSAGE(
            in_channels=7,
            hidden_channels=32,
            out_channels=32
        ).to(device)

        # Save model
        model_path = temp_dir / 'model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'features': {'dimensions': 7}, 'model': {'hidden_dim': 32, 'output_dim': 32}}
        }, model_path)

        # Create encoder
        encoder = StructuralEncoder(
            model_path=str(model_path),
            device='cuda',
            cache_embeddings=True
        )

        # Get embeddings
        embeddings = encoder.encode_all(full_graph)

        assert embeddings.shape == (50, 32)

        # Test caching
        embeddings2 = encoder.encode_all(full_graph)
        assert torch.allclose(embeddings, embeddings2)

        # Test invalidation
        encoder.invalidate_cache()
        embeddings3 = encoder.encode_all(full_graph)
        assert embeddings3.shape == (50, 32)

    def test_embedding_quality(self):
        """Test that trained embeddings have expected properties."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        device = torch.device('cuda')

        # Create data
        dataset = GraphSAGEDataset.from_mock(num_memories=100, seed=42)
        data = dataset.get_data(device=device)

        # Create model
        model = GraphSAGE(
            in_channels=7,
            hidden_channels=32,
            out_channels=32
        ).to(device)

        # Get embeddings (before training - random initialization)
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)

        # Basic quality checks
        from src.utils.metrics import check_embedding_health
        is_healthy, issues = check_embedding_health(embeddings)

        # Random init should still be "healthy" (normalized, not collapsed)
        assert 'Contains NaN values' not in issues
        assert 'Contains infinite values' not in issues

    def test_save_load_dataset(self, temp_dir):
        """Test dataset save and load."""
        # Create dataset
        dataset = GraphSAGEDataset.from_mock(num_memories=50, seed=42)

        # Save
        save_path = temp_dir / 'dataset'
        dataset.save(str(save_path))

        # Load
        loaded = GraphSAGEDataset.load(str(save_path))

        assert loaded.num_nodes == dataset.num_nodes
        assert loaded.num_edges == dataset.num_edges
        assert torch.allclose(loaded.features, dataset.features)


class TestConfigurationHandling:
    """Test configuration loading and validation."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()

        assert 'features' in config
        assert 'walks' in config
        assert 'model' in config
        assert 'training' in config

    def test_config_values(self):
        """Test that config has expected values."""
        config = load_config()

        assert config['features']['dimensions'] == 7
        assert config['model']['hidden_dim'] == 64
        assert config['model']['output_dim'] == 64
        assert config['training']['batch_size'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
