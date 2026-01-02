#!/usr/bin/env python3
"""
Data Generation Script.

This script generates training data for GraphSAGE:
1. Creates/loads graph
2. Extracts GraphSAGE view
3. Computes features
4. Generates random walks
5. Extracts positive pairs
6. Saves all data to disk

Usage:
    python scripts/generate_data.py --config config/default.yaml
    python scripts/generate_data.py --num-memories 500 --num-entities 100
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from config import load_config
from src.data import GraphSAGEDataset, GraphLoader
from src.data.view_extractor import GraphSAGEViewExtractor
from src.data.feature_extractor import MemoryFeatureExtractor
from src.walks import RandomWalkGenerator, CooccurrencePairSampler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate GraphSAGE training data')

    parser.add_argument(
        '--config', type=str, default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--graph-path', type=str, default=None,
        help='Path to existing graph file (JSON)'
    )
    parser.add_argument(
        '--num-memories', type=int, default=500,
        help='Number of memory nodes for synthetic graph'
    )
    parser.add_argument(
        '--num-entities', type=int, default=100,
        help='Number of entity nodes for synthetic graph'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed progress'
    )

    return parser.parse_args()


def main():
    """Main data generation function."""
    args = parse_args()

    print("=" * 60)
    print("GraphSAGE Data Generation")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    walk_config = config.get('walks', {})

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load or create graph
    print("\n[1/5] Loading/creating graph...")
    start_time = time.time()

    if args.graph_path:
        print(f"  Loading from: {args.graph_path}")
        full_graph = GraphLoader.from_file(args.graph_path)
    else:
        print(f"  Creating synthetic graph: {args.num_memories} memories, {args.num_entities} entities")
        full_graph = GraphLoader.create_mock(
            num_memories=args.num_memories,
            num_entities=args.num_entities,
            seed=args.seed
        )
        # Save synthetic graph
        full_graph.save(str(output_dir / 'full_graph.json'))

    print(f"  Done in {time.time() - start_time:.2f}s")

    # Step 2: Extract GraphSAGE view
    print("\n[2/5] Extracting GraphSAGE view...")
    start_time = time.time()

    view_extractor = GraphSAGEViewExtractor(full_graph)
    view = view_extractor.extract_with_undirected_edges()

    print(f"  Memory nodes: {view.num_nodes}")
    print(f"  Edges (undirected): {view.num_edges}")
    print(f"  Done in {time.time() - start_time:.2f}s")

    # Step 3: Compute features
    print("\n[3/5] Computing node features...")
    start_time = time.time()

    feature_extractor = MemoryFeatureExtractor(
        full_graph=full_graph,
        node_mapping=view.node_mapping,
        include_entity_features=config.get('features', {}).get('include_entity_features', True)
    )
    features = feature_extractor.extract(view.edge_index)

    print(f"  Feature shape: {features.shape}")
    print(f"  Done in {time.time() - start_time:.2f}s")

    # Feature statistics
    if args.verbose:
        stats = feature_extractor.get_statistics()
        print("\n  Feature statistics:")
        for name, s in stats.items():
            print(f"    {name}: mean={s['mean']:.3f}, std={s['std']:.3f}")

    # Step 4: Generate random walks
    print("\n[4/5] Generating random walks...")
    start_time = time.time()

    walk_length = walk_config.get('length', 80)
    walks_per_node = walk_config.get('per_node', 10)

    walker = RandomWalkGenerator(
        edge_index=view.edge_index,
        num_nodes=view.num_nodes,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        seed=walk_config.get('seed', args.seed)
    )

    walks = walker.generate_all_walks(verbose=args.verbose)

    print(f"  Generated {len(walks)} walks")
    avg_length = sum(len(w) for w in walks) / len(walks) if walks else 0
    print(f"  Average walk length: {avg_length:.1f}")
    print(f"  Done in {time.time() - start_time:.2f}s")

    # Step 5: Extract positive pairs
    print("\n[5/5] Extracting positive pairs...")
    start_time = time.time()

    context_window = walk_config.get('context_window', 10)
    pair_sampler = CooccurrencePairSampler(context_window=context_window)
    pairs = pair_sampler.extract_pairs(walks)

    print(f"  Extracted {len(pairs):,} positive pairs")
    print(f"  Done in {time.time() - start_time:.2f}s")

    # Save all data
    print("\n[Save] Saving processed data...")

    # Save features
    torch.save(features, output_dir / 'features.pt')

    # Save edge index
    torch.save(view.edge_index, output_dir / 'edge_index.pt')

    # Save mappings
    import json
    with open(output_dir / 'node_mapping.json', 'w') as f:
        json.dump({
            'node_mapping': view.node_mapping,
            'memory_ids': view.memory_ids
        }, f)

    # Save walks
    walks_dir = output_dir / 'walks'
    walks_dir.mkdir(exist_ok=True)
    with open(walks_dir / 'walks.json', 'w') as f:
        json.dump(walks, f)

    # Save pairs as tensors
    targets = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    torch.save({'targets': targets, 'contexts': contexts}, output_dir / 'pairs.pt')

    # Save metadata
    metadata = {
        'num_nodes': view.num_nodes,
        'num_edges': view.num_edges,
        'num_features': features.shape[1],
        'num_walks': len(walks),
        'num_pairs': len(pairs),
        'walk_length': walk_length,
        'walks_per_node': walks_per_node,
        'context_window': context_window,
        'seed': args.seed
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll data saved to: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.rglob('*'):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(output_dir)}: {size_kb:.1f} KB")

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
