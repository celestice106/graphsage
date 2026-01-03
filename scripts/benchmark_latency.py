#!/usr/bin/env python3
"""
Latency Benchmarking Script.

This script benchmarks inference latency of the GraphSAGE encoder:
- Single inference latency
- Batch inference throughput
- Memory usage
- Comparison with/without caching

Usage:
    python scripts/benchmark_latency.py --checkpoint exports/graphsage.pt
    python scripts/benchmark_latency.py --checkpoint exports/graphsage.pt --num-nodes 1000
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from config import load_config
from src.model import GraphSAGE
from src.data import GraphLoader
from src.inference import StructuralEncoder, benchmark_encoder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark GraphSAGE latency')

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--num-nodes', type=int, default=500,
        help='Number of nodes for synthetic graph'
    )
    parser.add_argument(
        '--num-entities', type=int, default=100,
        help='Number of entities for synthetic graph'
    )
    parser.add_argument(
        '--num-iterations', type=int, default=100,
        help='Number of benchmark iterations'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for benchmarking'
    )
    parser.add_argument(
        '--warmup', type=int, default=10,
        help='Number of warmup iterations'
    )

    return parser.parse_args()


def benchmark_raw_model(model, features, edge_index, num_iterations, warmup):
    """Benchmark raw model forward pass."""
    print("\n[1] Raw Model Forward Pass")
    print("-" * 40)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(features, edge_index)
    torch.cuda.synchronize() if features.is_cuda else None

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(features, edge_index)
        torch.cuda.synchronize() if features.is_cuda else None
        times.append(time.perf_counter() - start)

    times_ms = [t * 1000 for t in times]

    print(f"  Mean: {np.mean(times_ms):.3f} ms")
    print(f"  Std:  {np.std(times_ms):.3f} ms")
    print(f"  Min:  {np.min(times_ms):.3f} ms")
    print(f"  Max:  {np.max(times_ms):.3f} ms")
    print(f"  P50:  {np.percentile(times_ms, 50):.3f} ms")
    print(f"  P95:  {np.percentile(times_ms, 95):.3f} ms")
    print(f"  P99:  {np.percentile(times_ms, 99):.3f} ms")

    return {
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms),
        'p50_ms': np.percentile(times_ms, 50),
        'p95_ms': np.percentile(times_ms, 95),
        'p99_ms': np.percentile(times_ms, 99),
        'throughput_per_sec': 1000 / np.mean(times_ms)
    }


def benchmark_encoder_cached(encoder, full_graph, num_iterations):
    """Benchmark encoder with caching."""
    print("\n[2] Encoder with Caching")
    print("-" * 40)

    # First call (cache miss)
    start = time.perf_counter()
    _ = encoder.encode_all(full_graph, force_recompute=True)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000
    print(f"  First call (cache miss): {first_call_ms:.3f} ms")

    # Subsequent calls (cache hit)
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = encoder.encode_all(full_graph)
        times.append(time.perf_counter() - start)

    times_ms = [t * 1000 for t in times]

    print(f"  Cached calls mean: {np.mean(times_ms):.4f} ms")
    print(f"  Cache hit rate: {encoder.cache.hits / (encoder.cache.hits + encoder.cache.misses):.1%}")

    return {
        'first_call_ms': first_call_ms,
        'cached_mean_ms': np.mean(times_ms),
        'cache_speedup': first_call_ms / np.mean(times_ms)
    }


def benchmark_memory_usage(model, features, edge_index):
    """Benchmark GPU memory usage."""
    print("\n[3] Memory Usage")
    print("-" * 40)

    if not torch.cuda.is_available():
        print("  Skipping (CUDA not available)")
        return {}

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Get baseline memory
    baseline_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    # Forward pass
    with torch.no_grad():
        embeddings = model(features, edge_index)

    # Peak memory
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Embedding memory
    emb_mb = embeddings.element_size() * embeddings.numel() / (1024 * 1024)

    print(f"  Baseline memory: {baseline_mb:.2f} MB")
    print(f"  Peak memory: {peak_mb:.2f} MB")
    print(f"  Forward pass overhead: {peak_mb - baseline_mb:.2f} MB")
    print(f"  Embedding tensor: {emb_mb:.2f} MB")

    return {
        'baseline_mb': baseline_mb,
        'peak_mb': peak_mb,
        'overhead_mb': peak_mb - baseline_mb,
        'embedding_mb': emb_mb
    }


def benchmark_scaling(model, device, num_iterations):
    """Benchmark latency scaling with graph size."""
    print("\n[4] Scaling Analysis")
    print("-" * 40)

    sizes = [100, 500, 1000, 2000, 5000]
    results = []

    for num_nodes in sizes:
        # Create synthetic data
        features = torch.randn(num_nodes, model.in_channels, device=device)
        num_edges = min(num_nodes * 5, num_nodes * (num_nodes - 1) // 10)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(features, edge_index)
        torch.cuda.synchronize() if device.type == 'cuda' else None

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(features, edge_index)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(time.perf_counter() - start)

        mean_ms = np.mean(times) * 1000
        results.append((num_nodes, mean_ms))
        print(f"  {num_nodes:5d} nodes: {mean_ms:.3f} ms")

    return results


def main():
    """Main benchmark function."""
    args = parse_args()

    print("=" * 60)
    print("GraphSAGE Latency Benchmark")
    print("=" * 60)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint.get('config', load_config())
    model_config = config.get('model', {})
    feature_dim = config.get('features', {}).get('dimensions', 7)

    model = GraphSAGE(
        in_channels=feature_dim,
        hidden_channels=model_config.get('hidden_dim', 64),
        out_channels=model_config.get('output_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=0.0,
        normalize_output=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")

    # Create synthetic graph
    print(f"\nCreating synthetic graph ({args.num_nodes} nodes)...")
    full_graph = GraphLoader.create_mock(
        num_memories=args.num_nodes,
        num_entities=args.num_entities,
        seed=42
    )

    # Create features and edges for direct benchmark
    from src.data.view_extractor import GraphSAGEViewExtractor
    from src.data.feature_extractor import MemoryFeatureExtractor

    view_extractor = GraphSAGEViewExtractor(full_graph)
    view = view_extractor.extract_with_undirected_edges()

    feature_extractor = MemoryFeatureExtractor(
        full_graph=full_graph,
        node_mapping=view.node_mapping
    )
    features = feature_extractor.extract(view.edge_index).to(device)
    edge_index = view.edge_index.to(device)

    print(f"Graph: {view.num_nodes} nodes, {view.num_edges} edges")

    # Benchmark raw model
    raw_results = benchmark_raw_model(
        model, features, edge_index,
        args.num_iterations, args.warmup
    )

    # Benchmark with encoder + cache
    encoder = StructuralEncoder(
        model=model,
        device=str(device),
        cache_embeddings=True
    )
    cache_results = benchmark_encoder_cached(
        encoder, full_graph, args.num_iterations
    )

    # Memory usage
    memory_results = benchmark_memory_usage(model, features, edge_index)

    # Scaling analysis
    scaling_results = benchmark_scaling(model, device, min(args.num_iterations, 20))

    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    print(f"\nRaw model inference:")
    print(f"  Mean latency: {raw_results['mean_ms']:.3f} ms")
    print(f"  Throughput: {raw_results['throughput_per_sec']:.1f} inferences/sec")

    print(f"\nWith caching:")
    print(f"  First call: {cache_results['first_call_ms']:.3f} ms")
    print(f"  Cached: {cache_results['cached_mean_ms']:.4f} ms")
    print(f"  Speedup: {cache_results['cache_speedup']:.0f}x")

    # Target check
    target_ms = 5.0
    print(f"\n{'✓' if raw_results['mean_ms'] < target_ms else '✗'} Target latency (<{target_ms}ms): {raw_results['mean_ms']:.3f} ms")

    print("=" * 60)


if __name__ == '__main__':
    main()
