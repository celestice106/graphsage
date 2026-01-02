"""
Evaluation Metrics Module.

This module provides metrics for evaluating the quality of learned embeddings:
- Neighbor similarity: Do connected nodes have similar embeddings?
- Link prediction: Can we predict edges from embeddings?
- Embedding statistics: Distribution and quality checks
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_neighbor_similarity(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    sample_size: Optional[int] = 1000
) -> Dict:
    """
    Compute similarity statistics between connected nodes.

    Connected nodes should have higher similarity than random pairs
    if the embeddings capture graph structure.

    Args:
        embeddings: Node embeddings [num_nodes, dim]
        edge_index: Edge indices [2, num_edges]
        sample_size: Number of edges to sample (None = all)

    Returns:
        Dictionary with similarity statistics
    """
    if edge_index.numel() == 0:
        return {
            'neighbor_sim_mean': 0.0,
            'neighbor_sim_std': 0.0,
            'random_sim_mean': 0.0,
            'random_sim_std': 0.0,
            'sim_gap': 0.0
        }

    num_nodes = embeddings.shape[0]
    num_edges = edge_index.shape[1]

    # Sample edges if too many
    if sample_size is not None and num_edges > sample_size:
        idx = torch.randperm(num_edges)[:sample_size]
        src = edge_index[0, idx]
        dst = edge_index[1, idx]
    else:
        src = edge_index[0]
        dst = edge_index[1]

    # Compute neighbor similarities (dot product of normalized embeddings)
    # Note: embeddings should already be L2 normalized from the model
    src_emb = embeddings[src]
    dst_emb = embeddings[dst]
    neighbor_sims = (src_emb * dst_emb).sum(dim=1)

    # Compute random pair similarities for comparison
    random_src = torch.randint(0, num_nodes, (len(src),), device=embeddings.device)
    random_dst = torch.randint(0, num_nodes, (len(dst),), device=embeddings.device)
    random_src_emb = embeddings[random_src]
    random_dst_emb = embeddings[random_dst]
    random_sims = (random_src_emb * random_dst_emb).sum(dim=1)

    return {
        'neighbor_sim_mean': neighbor_sims.mean().item(),
        'neighbor_sim_std': neighbor_sims.std().item(),
        'random_sim_mean': random_sims.mean().item(),
        'random_sim_std': random_sims.std().item(),
        'sim_gap': (neighbor_sims.mean() - random_sims.mean()).item()
    }


def evaluate_link_prediction(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    num_negative_samples: int = 1000
) -> Dict:
    """
    Evaluate embeddings on link prediction task.

    Uses dot product similarity as edge score and computes
    AUC-ROC and Average Precision.

    Args:
        embeddings: Node embeddings [num_nodes, dim]
        edge_index: Edge indices [2, num_edges]
        num_negative_samples: Number of negative edges to sample

    Returns:
        Dictionary with link prediction metrics
    """
    if edge_index.numel() == 0:
        return {
            'auc_roc': 0.5,
            'avg_precision': 0.5
        }

    num_nodes = embeddings.shape[0]
    num_edges = edge_index.shape[1]

    # Use all edges as positives (or sample if too many)
    max_positives = min(num_edges, num_negative_samples)
    if num_edges > max_positives:
        idx = torch.randperm(num_edges)[:max_positives]
        pos_edge_index = edge_index[:, idx]
    else:
        pos_edge_index = edge_index

    # Create positive edge set for negative sampling
    edge_set = set(
        (int(s), int(d))
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist())
    )

    # Sample negative edges
    neg_edges = []
    attempts = 0
    max_attempts = num_negative_samples * 10

    while len(neg_edges) < num_negative_samples and attempts < max_attempts:
        s = np.random.randint(0, num_nodes)
        d = np.random.randint(0, num_nodes)
        if s != d and (s, d) not in edge_set:
            neg_edges.append([s, d])
        attempts += 1

    if len(neg_edges) == 0:
        return {'auc_roc': 0.5, 'avg_precision': 0.5}

    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()

    # Compute scores
    # Positive scores
    pos_src_emb = embeddings[pos_edge_index[0]]
    pos_dst_emb = embeddings[pos_edge_index[1]]
    pos_scores = (pos_src_emb * pos_dst_emb).sum(dim=1)

    # Negative scores
    neg_src_emb = embeddings[neg_edge_index[0]]
    neg_dst_emb = embeddings[neg_edge_index[1]]
    neg_scores = (neg_src_emb * neg_dst_emb).sum(dim=1)

    # Combine for evaluation
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores))
    ])

    # Compute metrics
    auc_roc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    return {
        'auc_roc': float(auc_roc),
        'avg_precision': float(avg_precision),
        'num_pos': len(pos_scores),
        'num_neg': len(neg_scores)
    }


def compute_embedding_statistics(embeddings: torch.Tensor) -> Dict:
    """
    Compute statistics about embedding quality.

    Checks for common issues like:
    - All embeddings identical (collapse)
    - Very low variance (near-collapse)
    - Embeddings not normalized

    Args:
        embeddings: Node embeddings [num_nodes, dim]

    Returns:
        Dictionary with embedding statistics
    """
    num_nodes, dim = embeddings.shape

    # Basic statistics
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)

    # Check normalization
    norms = embeddings.norm(dim=1)
    is_normalized = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Check for collapse (all embeddings similar)
    # Compute pairwise distances for sample
    sample_size = min(100, num_nodes)
    sample_idx = torch.randperm(num_nodes)[:sample_size]
    sample = embeddings[sample_idx]

    # Pairwise cosine similarities
    sample_normalized = F.normalize(sample, dim=1)
    sim_matrix = sample_normalized @ sample_normalized.t()
    # Exclude diagonal
    mask = ~torch.eye(sample_size, dtype=torch.bool, device=embeddings.device)
    off_diagonal_sims = sim_matrix[mask]

    mean_similarity = off_diagonal_sims.mean().item()
    max_similarity = off_diagonal_sims.max().item()

    # Collapse detection
    is_collapsed = mean_similarity > 0.99  # Almost all pairs very similar

    return {
        'num_nodes': num_nodes,
        'embedding_dim': dim,
        'mean_norm': norms.mean().item(),
        'std_norm': norms.std().item(),
        'is_normalized': is_normalized,
        'mean_per_dim': mean.mean().item(),
        'std_per_dim': std.mean().item(),
        'mean_pairwise_similarity': mean_similarity,
        'max_pairwise_similarity': max_similarity,
        'is_collapsed': is_collapsed
    }


def evaluate_embeddings(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor
) -> Dict:
    """
    Comprehensive embedding evaluation.

    Combines all metrics into a single evaluation.

    Args:
        embeddings: Node embeddings
        edge_index: Edge indices

    Returns:
        Dictionary with all metrics
    """
    results = {}

    # Embedding quality
    emb_stats = compute_embedding_statistics(embeddings)
    results['embedding_stats'] = emb_stats

    # Neighbor similarity
    neighbor_sim = compute_neighbor_similarity(embeddings, edge_index)
    results['neighbor_similarity'] = neighbor_sim

    # Link prediction
    link_pred = evaluate_link_prediction(embeddings, edge_index)
    results['link_prediction'] = link_pred

    # Summary
    results['summary'] = {
        'is_normalized': emb_stats['is_normalized'],
        'is_collapsed': emb_stats['is_collapsed'],
        'sim_gap': neighbor_sim['sim_gap'],
        'auc_roc': link_pred['auc_roc']
    }

    return results


def check_embedding_health(embeddings: torch.Tensor) -> Tuple[bool, List[str]]:
    """
    Quick health check for embeddings.

    Returns:
        Tuple of (is_healthy, list_of_issues)
    """
    issues = []

    stats = compute_embedding_statistics(embeddings)

    if not stats['is_normalized']:
        issues.append("Embeddings are not L2 normalized")

    if stats['is_collapsed']:
        issues.append("Embeddings have collapsed (all similar)")

    if stats['std_per_dim'] < 0.01:
        issues.append("Very low embedding variance")

    if torch.isnan(embeddings).any():
        issues.append("Contains NaN values")

    if torch.isinf(embeddings).any():
        issues.append("Contains infinite values")

    return len(issues) == 0, issues
