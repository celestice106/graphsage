#!/usr/bin/env python3
"""
Evaluation Script.

This script evaluates trained GraphSAGE embeddings:
- Neighbor similarity analysis
- Link prediction evaluation
- Embedding quality checks
- Visualization generation

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/model_best.pt
    python scripts/evaluate.py --checkpoint checkpoints/model_best.pt --visualize
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json

from config import load_config
from src.model import ProductionGraphSAGE
from src.utils.metrics import (
    evaluate_embeddings,
    compute_neighbor_similarity,
    evaluate_link_prediction,
    check_embedding_health
)
from src.utils.visualization import (
    plot_embeddings_tsne,
    plot_similarity_distribution,
    plot_degree_distribution
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate GraphSAGE embeddings')

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='Directory with processed data'
    )
    parser.add_argument(
        '--output-dir', type=str, default='evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for evaluation'
    )

    return parser.parse_args()


def load_model_and_data(checkpoint_path: str, data_dir: Path, device: torch.device):
    """Load trained model and data."""
    print("Loading model and data...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = load_config()

    model_config = config.get('model', {})
    feature_dim = config.get('features', {}).get('dimensions', 7)

    # Create model
    model = ProductionGraphSAGE(
        in_channels=feature_dim,
        hidden_channels=model_config.get('hidden_dim', 64),
        out_channels=model_config.get('output_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=0.0,  # No dropout for evaluation
        normalize_output=model_config.get('normalize_output', True)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load data
    features = torch.load(data_dir / 'features.pt').to(device)
    edge_index = torch.load(data_dir / 'edge_index.pt').to(device)

    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    return model, features, edge_index, metadata, config


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 60)
    print("GraphSAGE Evaluation")
    print("=" * 60)

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, features, edge_index, metadata, config = load_model_and_data(
        args.checkpoint, data_dir, device
    )

    print(f"\nGraph statistics:")
    print(f"  Nodes: {metadata['num_nodes']}")
    print(f"  Edges: {metadata['num_edges']}")

    # Compute embeddings
    print("\nComputing embeddings...")
    with torch.no_grad():
        embeddings = model(features, edge_index)

    print(f"  Embedding shape: {embeddings.shape}")

    # Health check
    print("\n[1] Embedding Health Check")
    print("-" * 40)
    is_healthy, issues = check_embedding_health(embeddings)

    if is_healthy:
        print("  ✓ Embeddings are healthy")
    else:
        print("  ✗ Issues detected:")
        for issue in issues:
            print(f"    - {issue}")

    # Neighbor similarity
    print("\n[2] Neighbor Similarity Analysis")
    print("-" * 40)
    sim_results = compute_neighbor_similarity(embeddings, edge_index)

    print(f"  Connected pairs similarity: {sim_results['neighbor_sim_mean']:.4f} ± {sim_results['neighbor_sim_std']:.4f}")
    print(f"  Random pairs similarity: {sim_results['random_sim_mean']:.4f} ± {sim_results['random_sim_std']:.4f}")
    print(f"  Similarity gap: {sim_results['sim_gap']:.4f}")

    if sim_results['sim_gap'] > 0.1:
        print("  ✓ Good: Connected pairs are more similar than random")
    else:
        print("  ⚠ Warning: Small similarity gap suggests embeddings may not capture structure well")

    # Link prediction
    print("\n[3] Link Prediction Evaluation")
    print("-" * 40)
    lp_results = evaluate_link_prediction(embeddings, edge_index)

    print(f"  AUC-ROC: {lp_results['auc_roc']:.4f}")
    print(f"  Average Precision: {lp_results['avg_precision']:.4f}")

    if lp_results['auc_roc'] > 0.75:
        print("  ✓ Good: Strong link prediction performance")
    elif lp_results['auc_roc'] > 0.6:
        print("  ⚠ Moderate: Acceptable link prediction")
    else:
        print("  ✗ Poor: Weak link prediction suggests embedding issues")

    # Full evaluation
    print("\n[4] Detailed Evaluation")
    print("-" * 40)
    full_results = evaluate_embeddings(embeddings, edge_index)

    emb_stats = full_results['embedding_stats']
    print(f"  Mean embedding norm: {emb_stats['mean_norm']:.4f}")
    print(f"  Mean pairwise similarity: {emb_stats['mean_pairwise_similarity']:.4f}")
    print(f"  Embeddings normalized: {emb_stats['is_normalized']}")
    print(f"  Embeddings collapsed: {emb_stats['is_collapsed']}")

    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'metadata': metadata,
        'health': {'is_healthy': is_healthy, 'issues': issues},
        'neighbor_similarity': sim_results,
        'link_prediction': lp_results,
        'embedding_stats': emb_stats
    }

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'evaluation_results.json'}")

    # Visualization
    if args.visualize:
        print("\n[5] Generating Visualizations")
        print("-" * 40)

        # t-SNE plot
        print("  Creating t-SNE visualization...")
        plot_embeddings_tsne(
            embeddings,
            output_path=str(output_dir / 'embeddings_tsne.png'),
            show=False
        )

        # Similarity distribution
        print("  Creating similarity distribution plot...")
        plot_similarity_distribution(
            embeddings,
            edge_index,
            output_path=str(output_dir / 'similarity_distribution.png'),
            show=False
        )

        # Degree distribution
        print("  Creating degree distribution plot...")
        plot_degree_distribution(
            edge_index,
            metadata['num_nodes'],
            output_path=str(output_dir / 'degree_distribution.png'),
            show=False
        )

        print(f"\n  Visualizations saved to: {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    quality_score = 0
    if is_healthy:
        quality_score += 1
    if sim_results['sim_gap'] > 0.1:
        quality_score += 1
    if lp_results['auc_roc'] > 0.75:
        quality_score += 1
    if not emb_stats['is_collapsed']:
        quality_score += 1

    print(f"Quality Score: {quality_score}/4")

    if quality_score >= 3:
        print("✓ Embeddings are production-ready")
    elif quality_score >= 2:
        print("⚠ Embeddings are acceptable but could be improved")
    else:
        print("✗ Embeddings need improvement - consider retraining")

    print("=" * 60)


if __name__ == '__main__':
    main()
