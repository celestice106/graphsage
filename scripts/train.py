#!/usr/bin/env python3
"""
GraphSAGE Training Script.

This script trains the GraphSAGE model using the skip-gram objective
with random walk co-occurrence pairs.

Usage:
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --data-dir data/processed --epochs 100

For Google Colab, use the train_colab.ipynb notebook instead.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json

from config import load_config
from src.data import GraphSAGEDataset
from src.walks import RandomWalkGenerator, CooccurrencePairSampler, DegreeBiasedNegativeSampler
from src.model import GraphSAGE, SkipGramLoss
from src.training import GraphSAGETrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GraphSAGE model')

    parser.add_argument(
        '--config', type=str, default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='Directory with processed data'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for training (cuda or cpu)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--generate-walks', action='store_true',
        help='Generate walks instead of loading from disk'
    )

    return parser.parse_args()


def load_data(data_dir: Path, device: torch.device):
    """Load preprocessed data."""
    print("Loading data...")

    # Load features
    features = torch.load(data_dir / 'features.pt')

    # Load edge index
    edge_index = torch.load(data_dir / 'edge_index.pt')

    # Load pairs
    pairs_data = torch.load(data_dir / 'pairs.pt')
    targets = pairs_data['targets']
    contexts = pairs_data['contexts']
    pairs = list(zip(targets.tolist(), contexts.tolist()))

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"  Nodes: {metadata['num_nodes']}")
    print(f"  Edges: {metadata['num_edges']}")
    print(f"  Features: {features.shape}")
    print(f"  Pairs: {len(pairs):,}")

    return features.to(device), edge_index.to(device), pairs, metadata


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("GraphSAGE Training")
    print("=" * 60)

    # Setup device
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr

    # Load data
    data_dir = Path(args.data_dir)
    features, edge_index, pairs, metadata = load_data(data_dir, device)

    num_nodes = metadata['num_nodes']

    # Generate walks if requested (instead of using pre-computed pairs)
    if args.generate_walks:
        print("\nGenerating random walks...")
        walk_config = config.get('walks', {})

        walker = RandomWalkGenerator(
            edge_index=edge_index.cpu(),  # Walker works on CPU
            num_nodes=num_nodes,
            walk_length=walk_config.get('length', 80),
            walks_per_node=walk_config.get('per_node', 10),
            seed=walk_config.get('seed', 42)
        )
        walks = walker.generate_all_walks(verbose=True)

        pair_sampler = CooccurrencePairSampler(
            context_window=walk_config.get('context_window', 10)
        )
        pairs = pair_sampler.extract_pairs(walks)
        print(f"Generated {len(pairs):,} pairs")

    # Setup negative sampler
    neg_config = config.get('negatives', {})
    neg_sampler = DegreeBiasedNegativeSampler(
        edge_index=edge_index,
        num_nodes=num_nodes,
        exponent=neg_config.get('exponent', 0.75),
        device=device
    )

    # Create model
    model_config = config.get('model', {})
    feature_dim = config.get('features', {}).get('dimensions', 7)

    model = GraphSAGE(
        in_channels=feature_dim,
        hidden_channels=model_config.get('hidden_dim', 64),
        out_channels=model_config.get('output_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.3),
        normalize_output=model_config.get('normalize_output', True)
    ).to(device)

    print(f"\nModel: {model.count_parameters():,} parameters")

    # Create trainer
    trainer = GraphSAGETrainer(
        model=model,
        features=features,
        edge_index=edge_index,
        positive_pairs=pairs,
        negative_sampler=neg_sampler,
        config=config
    )

    # Resume from checkpoint if provided
    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train
    train_config = config.get('training', {})
    num_epochs = train_config.get('epochs', 100)

    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()

    best_loss = trainer.train(num_epochs=num_epochs)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")

    # Save final model
    export_dir = Path(config.get('paths', {}).get('exports', 'exports'))
    export_dir.mkdir(parents=True, exist_ok=True)

    final_path = export_dir / 'graphsage_final.pt'
    trainer.save_checkpoint(str(final_path))
    print(f"Final model saved to: {final_path}")

    # Evaluate final embeddings
    print("\nEvaluating final embeddings...")
    embeddings = trainer.get_embeddings()

    from src.utils.metrics import evaluate_embeddings
    eval_results = evaluate_embeddings(embeddings, edge_index)

    print("\nEvaluation Results:")
    print(f"  Neighbor similarity gap: {eval_results['neighbor_similarity']['sim_gap']:.4f}")
    print(f"  Link prediction AUC: {eval_results['link_prediction']['auc_roc']:.4f}")
    print(f"  Embeddings collapsed: {eval_results['embedding_stats']['is_collapsed']}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
