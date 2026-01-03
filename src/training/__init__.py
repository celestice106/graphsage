"""
Training Module for GraphSAGE.

This module implements the complete training pipeline including:
- Batch generation from positive pairs
- Training loop with validation
- Early stopping and checkpointing
- Logging and metrics tracking

Components:
    GraphSAGETrainer: Main trainer class
    BatchGenerator: Generate training batches
    EarlyStopping: Patience-based early stopping
    TrainingLogger: Logging and metrics tracking

Example:
    >>> from src.training import GraphSAGETrainer
    >>> from src.model import GraphSAGE, SkipGramLoss
    >>>
    >>> model = GraphSAGE()
    >>> trainer = GraphSAGETrainer(
    ...     model=model,
    ...     features=features,
    ...     edge_index=edge_index,
    ...     positive_pairs=pairs,
    ...     negative_sampler=neg_sampler,
    ...     config=config
    ... )
    >>> trainer.train(num_epochs=100)
"""

from .trainer import GraphSAGETrainer
from .batch_generator import BatchGenerator, TrainingData
from .callbacks import EarlyStopping, TrainingLogger, ModelCheckpoint

__all__ = [
    'GraphSAGETrainer',
    'BatchGenerator',
    'TrainingData',
    'EarlyStopping',
    'TrainingLogger',
    'ModelCheckpoint',
]
