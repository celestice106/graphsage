"""
GraphSAGE Trainer Module.

This module implements the complete training pipeline for GraphSAGE
with skip-gram objective. It handles:
- Training loop with batch iteration
- Validation evaluation
- Early stopping and checkpointing
- Gradient clipping
- GPU memory management

Design Decisions:
- GPU-only training (no CPU fallback for performance)
- Full-batch graph forward pass (memory efficient for our graph sizes)
- On-the-fly negative sampling (fresh negatives each batch)
- No AMP by default (stability over speed for quality focus)
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional, Any
import random
import time
from pathlib import Path

from ..model.graphsage import ProductionGraphSAGE
from ..model.loss import SkipGramLoss
from ..walks import DegreeBiasedNegativeSampler
from .batch_generator import BatchGenerator, split_pairs_train_val
from .callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger


class GraphSAGETrainer:
    """
    Complete training pipeline for GraphSAGE with skip-gram objective.

    This trainer implements the GDS-style random walk co-occurrence approach:
    1. Forward pass: Compute embeddings for all nodes
    2. Extract embeddings for batch (targets, contexts, negatives)
    3. Compute skip-gram loss
    4. Backward pass and optimization

    Key Features:
    - GPU-only training for maximum performance
    - Gradient clipping for stability
    - Early stopping to prevent overfitting
    - Automatic checkpointing of best model
    - Detailed logging and metrics

    Example:
        >>> from src.training import GraphSAGETrainer
        >>> from src.model import ProductionGraphSAGE
        >>> from config import load_config
        >>>
        >>> config = load_config()
        >>> model = ProductionGraphSAGE().cuda()
        >>>
        >>> trainer = GraphSAGETrainer(
        ...     model=model,
        ...     features=features.cuda(),
        ...     edge_index=edge_index.cuda(),
        ...     positive_pairs=pairs,
        ...     negative_sampler=neg_sampler,
        ...     config=config
        ... )
        >>>
        >>> best_loss = trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        model: ProductionGraphSAGE,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        positive_pairs: List[Tuple[int, int]],
        negative_sampler: DegreeBiasedNegativeSampler,
        config: Dict[str, Any],
        val_pairs: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Initialize trainer.

        Args:
            model: GraphSAGE model (should already be on GPU)
            features: Node features [num_nodes, num_features] (on GPU)
            edge_index: Edge indices [2, num_edges] (on GPU)
            positive_pairs: List of (target, context) tuples
            negative_sampler: Sampler for negative nodes
            config: Training configuration dictionary
            val_pairs: Optional validation pairs (if None, split from positive_pairs)
        """
        # Verify GPU placement
        self.device = next(model.parameters()).device
        assert self.device.type == 'cuda', "Model must be on GPU for training"
        assert features.device == self.device, "Features must be on same device as model"
        assert edge_index.device == self.device, "Edge index must be on same device as model"

        self.model = model
        self.features = features
        self.edge_index = edge_index
        self.config = config

        # Extract training config
        train_config = config.get('training', {})
        self.learning_rate = train_config.get('learning_rate', 0.001)
        self.weight_decay = train_config.get('weight_decay', 0.0)
        self.batch_size = train_config.get('batch_size', 512)
        self.gradient_clip = train_config.get('gradient_clip', 1.0)
        self.num_negatives = config.get('negatives', {}).get('per_positive', 5)

        # Split train/val if validation pairs not provided
        val_config = config.get('validation', {})
        val_fraction = val_config.get('val_fraction', 0.1)

        if val_pairs is None and val_fraction > 0:
            train_pairs, val_pairs = split_pairs_train_val(
                positive_pairs, val_fraction,
                seed=train_config.get('seed', 42)
            )
        else:
            train_pairs = positive_pairs

        self.train_pairs = train_pairs
        self.val_pairs = val_pairs

        # Setup batch generator
        self.batch_generator = BatchGenerator(
            positive_pairs=train_pairs,
            negative_sampler=negative_sampler,
            batch_size=self.batch_size,
            num_negatives=self.num_negatives,
            device=self.device,
            drop_last=False
        )

        # Validation batch generator (if applicable)
        self.val_batch_generator = None
        if val_pairs:
            self.val_batch_generator = BatchGenerator(
                positive_pairs=val_pairs,
                negative_sampler=negative_sampler,
                batch_size=self.batch_size * 2,  # Can use larger batch for val
                num_negatives=self.num_negatives,
                device=self.device
            )

        # Setup optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Setup loss function
        self.loss_fn = SkipGramLoss()

        # Setup callbacks
        paths_config = config.get('paths', {})
        self.checkpoint = ModelCheckpoint(
            save_dir=paths_config.get('checkpoints', 'checkpoints'),
            save_best=True,
            save_every=train_config.get('checkpoint_every', 10)
        )

        self.early_stopping = EarlyStopping(
            patience=train_config.get('early_stopping_patience', 10),
            min_delta=train_config.get('min_delta', 0.0001)
        )

        self.logger = TrainingLogger(
            log_dir=paths_config.get('logs', 'logs'),
            log_every=train_config.get('log_every', 10)
        )

        # Tracking
        self.best_loss = float('inf')
        self.current_epoch = 0

        # Print setup summary
        self._print_setup_summary()

    def _print_setup_summary(self):
        """Print training setup summary."""
        print("=" * 60)
        print("GraphSAGE Training Setup")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training pairs: {len(self.train_pairs):,}")
        if self.val_pairs:
            print(f"Validation pairs: {len(self.val_pairs):,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batches per epoch: {len(self.batch_generator)}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Negative samples: {self.num_negatives}")
        print("=" * 60)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_pos_loss = 0.0
        total_neg_loss = 0.0
        num_batches = 0

        for batch in self.batch_generator:
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass: compute all node embeddings
            # This is done once per batch, then we index into it
            embeddings = self.model(self.features, self.edge_index)

            # Compute loss using batch indices
            loss, details = self.loss_fn.forward_with_details(
                embeddings,
                batch.targets,
                batch.contexts,
                batch.negatives
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            total_loss += details['total_loss']
            total_pos_loss += details['pos_loss']
            total_neg_loss += details['neg_loss']
            num_batches += 1

        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'pos_loss': total_pos_loss / num_batches,
            'neg_loss': total_neg_loss / num_batches,
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_batch_generator is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_pos_prob = 0.0
        total_neg_prob = 0.0
        num_batches = 0

        # Compute embeddings once for validation
        embeddings = self.model(self.features, self.edge_index)

        for batch in self.val_batch_generator:
            loss, details = self.loss_fn.forward_with_details(
                embeddings,
                batch.targets,
                batch.contexts,
                batch.negatives
            )

            total_loss += details['total_loss']
            total_pos_prob += details['pos_prob']
            total_neg_prob += details['neg_prob']
            num_batches += 1

        metrics = {
            'loss': total_loss / num_batches,
            'pos_prob': total_pos_prob / num_batches,  # Should be high
            'neg_prob': total_neg_prob / num_batches,  # Should be low
        }

        return metrics

    def train(self, num_epochs: int) -> float:
        """
        Full training loop.

        Args:
            num_epochs: Maximum number of epochs

        Returns:
            Best validation loss achieved
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print("-" * 60)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.logger.start_epoch(epoch)

            # Train one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # End epoch timing
            self.logger.end_epoch()

            # Log metrics
            self.logger.log_epoch(epoch, train_metrics, val_metrics)

            # Determine metric to monitor (prefer val_loss if available)
            monitor_value = val_metrics.get('loss', train_metrics['loss'])

            # Checkpointing
            self.checkpoint.on_epoch_end(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                value=monitor_value,
                extra_state={
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': self.config
                }
            )

            # Update best loss
            if monitor_value < self.best_loss:
                self.best_loss = monitor_value

            # Early stopping
            if self.early_stopping(monitor_value, epoch):
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best loss: {self.best_loss:.4f} at epoch {self.early_stopping.best_epoch}")
                break

        # Save final logs
        self.logger.save_final({
            'best_loss': self.best_loss,
            'best_epoch': self.early_stopping.best_epoch,
            'stopped_epoch': self.current_epoch
        })

        print("-" * 60)
        print(f"Training complete. Best loss: {self.best_loss:.4f}")

        return self.best_loss

    def get_embeddings(self) -> torch.Tensor:
        """
        Get current node embeddings.

        Returns:
            Embeddings tensor [num_nodes, embedding_dim]
        """
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(self.features, self.edge_index)
        return embeddings

    def save_checkpoint(self, path: str):
        """
        Save current state to checkpoint.

        Args:
            path: Output path
        """
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load state from checkpoint.

        Args:
            path: Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))


def train_graphsage(
    dataset,
    walks: List[List[int]],
    config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Tuple[ProductionGraphSAGE, torch.Tensor]:
    """
    High-level function to train GraphSAGE from dataset and walks.

    This is the recommended entry point for training.

    Args:
        dataset: GraphSAGEDataset instance
        walks: Pre-generated random walks
        config: Configuration dictionary
        device: Device to use (defaults to cuda)

    Returns:
        Tuple of (trained_model, embeddings)
    """
    from ..walks import CooccurrencePairSampler, DegreeBiasedNegativeSampler
    from ..model import ProductionGraphSAGE

    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', "GPU required for training"

    print(f"Using device: {device}")

    # Get data
    data = dataset.get_data(device=device)

    # Extract pairs from walks
    walk_config = config.get('walks', {})
    pair_sampler = CooccurrencePairSampler(
        context_window=walk_config.get('context_window', 10)
    )
    pairs = pair_sampler.extract_pairs(walks)
    print(f"Extracted {len(pairs):,} positive pairs from {len(walks):,} walks")

    # Setup negative sampler
    neg_config = config.get('negatives', {})
    neg_sampler = DegreeBiasedNegativeSampler(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        exponent=neg_config.get('exponent', 0.75),
        device=device
    )

    # Create model
    model_config = config.get('model', {})
    model = ProductionGraphSAGE(
        in_channels=config.get('features', {}).get('dimensions', 7),
        hidden_channels=model_config.get('hidden_dim', 64),
        out_channels=model_config.get('output_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.3),
        normalize_output=model_config.get('normalize_output', True)
    ).to(device)

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
    train_config = config.get('training', {})
    trainer.train(num_epochs=train_config.get('epochs', 100))

    # Get final embeddings
    embeddings = trainer.get_embeddings()

    return model, embeddings
