"""
Training Callbacks Module.

This module implements callbacks for training control and monitoring:
- EarlyStopping: Stop training when validation loss plateaus
- ModelCheckpoint: Save best model checkpoints
- TrainingLogger: Log metrics and training progress
"""

import torch
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a metric (usually validation loss) and stops training
    when no improvement is seen for 'patience' epochs.

    Example:
        >>> early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        >>>
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stopping(val_loss):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for metrics (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.stopped_epoch = 0

    def __call__(self, value: float, epoch: int = 0) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True
            return False

    def reset(self):
        """Reset early stopping state."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.stopped_epoch = 0

    def state_dict(self) -> Dict:
        """Get state for serialization."""
        return {
            'best_value': self.best_value,
            'counter': self.counter,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch
        }

    def load_state_dict(self, state: Dict):
        """Load state from dict."""
        self.best_value = state['best_value']
        self.counter = state['counter']
        self.best_epoch = state['best_epoch']
        self.stopped_epoch = state['stopped_epoch']


class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Saves the model whenever a monitored metric improves,
    and optionally saves periodic checkpoints.

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     save_dir='checkpoints',
        ...     save_best=True,
        ...     save_every=10
        ... )
        >>>
        >>> for epoch in range(100):
        ...     loss = train_epoch()
        ...     checkpoint.on_epoch_end(epoch, model, optimizer, loss)
    """

    def __init__(
        self,
        save_dir: str = 'checkpoints',
        save_best: bool = True,
        save_every: Optional[int] = None,
        mode: str = 'min',
        filename_prefix: str = 'model'
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_every: Save every N epochs (None = only best)
            mode: 'min' or 'max' for comparison
            filename_prefix: Prefix for checkpoint files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_best = save_best
        self.save_every = save_every
        self.mode = mode
        self.filename_prefix = filename_prefix

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        value: float,
        extra_state: Optional[Dict] = None
    ) -> bool:
        """
        Called at end of each epoch.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            value: Metric value for comparison
            extra_state: Additional state to save

        Returns:
            True if a checkpoint was saved
        """
        saved = False

        # Check if this is the best model
        if self.mode == 'min':
            is_best = value < self.best_value
        else:
            is_best = value > self.best_value

        if is_best:
            self.best_value = value
            self.best_epoch = epoch

            if self.save_best:
                self._save_checkpoint(
                    epoch, model, optimizer, value,
                    f'{self.filename_prefix}_best.pt',
                    extra_state
                )
                saved = True

        # Periodic saving
        if self.save_every and (epoch + 1) % self.save_every == 0:
            self._save_checkpoint(
                epoch, model, optimizer, value,
                f'{self.filename_prefix}_epoch{epoch}.pt',
                extra_state
            )
            saved = True

        return saved

    def _save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        value: float,
        filename: str,
        extra_state: Optional[Dict] = None
    ):
        """Save checkpoint to file."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'value': value,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat()
        }

        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, self.save_dir / filename)

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load best checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to restore

        Returns:
            Checkpoint dictionary
        """
        path = self.save_dir / f'{self.filename_prefix}_best.pt'
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint


class TrainingLogger:
    """
    Log training metrics and progress.

    Provides:
    - Console logging
    - JSON metrics file
    - Training time tracking

    Example:
        >>> logger = TrainingLogger(log_dir='logs', log_every=10)
        >>>
        >>> for epoch in range(100):
        ...     metrics = train_epoch()
        ...     logger.log_epoch(epoch, metrics)
        >>>
        >>> logger.save_final()
    """

    def __init__(
        self,
        log_dir: str = 'logs',
        log_every: int = 10,
        verbose: bool = True
    ):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            log_every: Print to console every N epochs
            verbose: Whether to print to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_every = log_every
        self.verbose = verbose

        # Tracking
        self.epoch_metrics: List[Dict] = []
        self.batch_metrics: List[Dict] = []
        self.start_time = time.time()
        self.epoch_times: List[float] = []

        # Current epoch tracking
        self.current_epoch = 0
        self.epoch_start_time = None

    def start_epoch(self, epoch: int):
        """Mark start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def end_epoch(self):
        """Mark end of an epoch."""
        if self.epoch_start_time is not None:
            elapsed = time.time() - self.epoch_start_time
            self.epoch_times.append(elapsed)

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Training metrics
            val_metrics: Optional validation metrics
        """
        record = {
            'epoch': epoch,
            'timestamp': time.time() - self.start_time,
            **{f'train_{k}': v for k, v in metrics.items()}
        }

        if val_metrics:
            record.update({f'val_{k}': v for k, v in val_metrics.items()})

        self.epoch_metrics.append(record)

        # Console logging
        if self.verbose and epoch % self.log_every == 0:
            self._print_epoch(epoch, metrics, val_metrics)

    def log_batch(
        self,
        epoch: int,
        batch: int,
        metrics: Dict[str, float]
    ):
        """Log metrics for a batch (optional, for detailed tracking)."""
        record = {
            'epoch': epoch,
            'batch': batch,
            'timestamp': time.time() - self.start_time,
            **metrics
        }
        self.batch_metrics.append(record)

    def _print_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]]
    ):
        """Print epoch summary to console."""
        parts = [f"Epoch {epoch:4d}"]

        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")

        if val_metrics:
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    parts.append(f"val_{key}: {value:.4f}")

        if self.epoch_times:
            avg_time = sum(self.epoch_times[-10:]) / len(self.epoch_times[-10:])
            parts.append(f"({avg_time:.1f}s/epoch)")

        print(" | ".join(parts))

    def save_final(self, extra_info: Optional[Dict] = None):
        """
        Save final training log.

        Args:
            extra_info: Additional info to include
        """
        total_time = time.time() - self.start_time

        summary = {
            'total_epochs': len(self.epoch_metrics),
            'total_time_seconds': total_time,
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times)
                             if self.epoch_times else 0,
            'final_metrics': self.epoch_metrics[-1] if self.epoch_metrics else {},
            'best_loss': min(m.get('train_loss', float('inf'))
                           for m in self.epoch_metrics) if self.epoch_metrics else None,
        }

        if extra_info:
            summary.update(extra_info)

        # Save epoch metrics
        with open(self.log_dir / 'epoch_metrics.json', 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)

        # Save summary
        with open(self.log_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print(f"\nTraining complete in {total_time/60:.1f} minutes")
            print(f"Logs saved to {self.log_dir}")

    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        return [m.get(metric_name) for m in self.epoch_metrics
                if metric_name in m]


class ProgressBar:
    """
    Simple progress bar for batch iteration.

    Example:
        >>> pbar = ProgressBar(total=100, desc='Training')
        >>> for i in range(100):
        ...     pbar.update(1, loss=0.5)
        >>> pbar.close()
    """

    def __init__(
        self,
        total: int,
        desc: str = '',
        disable: bool = False
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of steps
            desc: Description prefix
            disable: Whether to disable display
        """
        self.total = total
        self.desc = desc
        self.disable = disable

        self.n = 0
        self.start_time = time.time()
        self.metrics: Dict[str, float] = {}

    def update(self, n: int = 1, **metrics):
        """Update progress."""
        self.n += n
        self.metrics.update(metrics)

        if not self.disable:
            self._display()

    def _display(self):
        """Display progress."""
        elapsed = time.time() - self.start_time
        rate = self.n / elapsed if elapsed > 0 else 0

        # Build progress string
        progress = f"\r{self.desc}: {self.n}/{self.total} [{elapsed:.0f}s, {rate:.1f}it/s]"

        for key, value in self.metrics.items():
            if isinstance(value, float):
                progress += f" {key}={value:.4f}"

        print(progress, end='', flush=True)

    def close(self):
        """Close progress bar."""
        if not self.disable:
            print()  # Newline

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
