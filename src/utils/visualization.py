"""
Visualization Module.

This module provides plotting utilities for:
- Training curves
- Embedding visualization (t-SNE, PCA)
- Graph statistics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


def plot_training_curves(
    metrics_path: str,
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training curves from logged metrics.

    Args:
        metrics_path: Path to epoch_metrics.json
        output_path: Optional path to save figure
        show: Whether to display plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    epochs = [m['epoch'] for m in metrics]
    train_loss = [m.get('train_loss', None) for m in metrics]
    val_loss = [m.get('val_loss', None) for m in metrics]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax = axes[0]
    ax.plot(epochs, train_loss, label='Train Loss', marker='.')
    if val_loss[0] is not None:
        ax.plot(epochs, val_loss, label='Val Loss', marker='.')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot pos/neg probabilities if available
    ax = axes[1]
    if 'val_pos_prob' in metrics[0]:
        pos_prob = [m.get('val_pos_prob', None) for m in metrics]
        neg_prob = [m.get('val_neg_prob', None) for m in metrics]
        ax.plot(epochs, pos_prob, label='Pos Prob', marker='.')
        ax.plot(epochs, neg_prob, label='Neg Prob', marker='.')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Probability')
        ax.set_title('Positive/Negative Classification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_embeddings_tsne(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None,
    perplexity: int = 30,
    show: bool = True,
    sample_size: int = 500
) -> None:
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings: Node embeddings [num_nodes, dim]
        labels: Optional node labels for coloring
        edge_index: Optional edges to draw
        output_path: Path to save figure
        perplexity: t-SNE perplexity parameter
        show: Whether to display
        sample_size: Max nodes to plot (for performance)
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("Install requirements: pip install matplotlib scikit-learn")
        return

    # Sample if too many nodes
    num_nodes = embeddings.shape[0]
    if num_nodes > sample_size:
        idx = torch.randperm(num_nodes)[:sample_size]
        embeddings = embeddings[idx]
        if labels is not None:
            labels = labels[idx]
        # Adjust edge_index (skip for simplicity)
        edge_index = None

    # Move to CPU and numpy
    emb_np = embeddings.cpu().numpy()

    # Run t-SNE
    print(f"Running t-SNE on {emb_np.shape[0]} embeddings...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(emb_np)-1),
                random_state=42, n_iter=1000)
    coords = tsne.fit_transform(emb_np)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw edges if provided
    if edge_index is not None and edge_index.numel() > 0:
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        for s, d in zip(src, dst):
            ax.plot([coords[s, 0], coords[d, 0]],
                   [coords[s, 1], coords[d, 1]],
                   'gray', alpha=0.1, linewidth=0.5)

    # Draw nodes
    if labels is not None:
        labels_np = labels.cpu().numpy()
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels_np,
                           cmap='tab10', alpha=0.7, s=30)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30)

    ax.set_title('t-SNE Embedding Visualization')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_degree_distribution(
    edge_index: torch.Tensor,
    num_nodes: int,
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot degree distribution histogram.

    Args:
        edge_index: Edge indices
        num_nodes: Number of nodes
        output_path: Path to save figure
        show: Whether to display
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    if edge_index.numel() == 0:
        print("No edges to plot")
        return

    # Compute degrees
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()
    out_deg = torch.bincount(src, minlength=num_nodes)
    in_deg = torch.bincount(dst, minlength=num_nodes)
    total_deg = out_deg + in_deg

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    ax.hist(total_deg.numpy(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title('Degree Distribution')
    ax.axvline(total_deg.float().mean().item(), color='red',
               linestyle='--', label=f'Mean: {total_deg.float().mean():.1f}')
    ax.legend()

    # Log-log plot (for power-law check)
    ax = axes[1]
    degrees, counts = np.unique(total_deg.numpy(), return_counts=True)
    ax.loglog(degrees[degrees > 0], counts[degrees > 0], 'o', alpha=0.7)
    ax.set_xlabel('Degree (log)')
    ax.set_ylabel('Count (log)')
    ax.set_title('Degree Distribution (log-log)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_similarity_distribution(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    output_path: Optional[str] = None,
    show: bool = True,
    sample_size: int = 1000
) -> None:
    """
    Plot distribution of neighbor vs random similarities.

    Args:
        embeddings: Node embeddings
        edge_index: Edge indices
        output_path: Path to save
        show: Whether to display
        sample_size: Number of pairs to sample
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    num_nodes = embeddings.shape[0]
    num_edges = edge_index.shape[1]

    # Sample edges
    if num_edges > sample_size:
        idx = torch.randperm(num_edges)[:sample_size]
        src = edge_index[0, idx]
        dst = edge_index[1, idx]
    else:
        src = edge_index[0]
        dst = edge_index[1]

    # Neighbor similarities
    neighbor_sims = (embeddings[src] * embeddings[dst]).sum(dim=1).cpu().numpy()

    # Random similarities
    rand_src = torch.randint(0, num_nodes, (len(src),))
    rand_dst = torch.randint(0, num_nodes, (len(src),))
    random_sims = (embeddings[rand_src] * embeddings[rand_dst]).sum(dim=1).cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(random_sims, bins=50, alpha=0.5, label='Random pairs', density=True)
    ax.hist(neighbor_sims, bins=50, alpha=0.5, label='Connected pairs', density=True)

    ax.axvline(random_sims.mean(), color='blue', linestyle='--',
               label=f'Random mean: {random_sims.mean():.3f}')
    ax.axvline(neighbor_sims.mean(), color='orange', linestyle='--',
               label=f'Neighbor mean: {neighbor_sims.mean():.3f}')

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Embedding Similarity Distribution')
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def create_training_report(
    log_dir: str,
    output_path: str
) -> None:
    """
    Create a comprehensive training report.

    Args:
        log_dir: Directory containing training logs
        output_path: Path to save report
    """
    log_dir = Path(log_dir)

    # Load metrics
    with open(log_dir / 'epoch_metrics.json', 'r') as f:
        metrics = json.load(f)

    with open(log_dir / 'training_summary.json', 'r') as f:
        summary = json.load(f)

    # Generate plots
    plot_training_curves(
        str(log_dir / 'epoch_metrics.json'),
        output_path=str(Path(output_path).parent / 'training_curves.png'),
        show=False
    )

    # Create text report
    report = []
    report.append("=" * 60)
    report.append("GraphSAGE Training Report")
    report.append("=" * 60)
    report.append("")
    report.append(f"Total epochs: {summary['total_epochs']}")
    report.append(f"Training time: {summary['total_time_seconds']/60:.1f} minutes")
    report.append(f"Best loss: {summary['best_loss']:.4f}")
    report.append(f"Best epoch: {summary.get('best_epoch', 'N/A')}")
    report.append("")
    report.append("Final Metrics:")
    for key, value in summary.get('final_metrics', {}).items():
        if isinstance(value, float):
            report.append(f"  {key}: {value:.4f}")
        else:
            report.append(f"  {key}: {value}")

    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(report_text)
