#!/usr/bin/env python3
"""
Model Export Script.

This script exports trained GraphSAGE models for production use:
- PyTorch format (for Python inference)
- TorchScript format (for C++ inference)
- ONNX format (for cross-platform deployment)

Usage:
    python scripts/export_model.py --checkpoint checkpoints/model_best.pt
    python scripts/export_model.py --checkpoint checkpoints/model_best.pt --format all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from config import load_config
from src.model import ProductionGraphSAGE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export GraphSAGE model')

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir', type=str, default='exports',
        help='Output directory'
    )
    parser.add_argument(
        '--format', type=str, default='pytorch',
        choices=['pytorch', 'torchscript', 'onnx', 'all'],
        help='Export format'
    )
    parser.add_argument(
        '--num-nodes', type=int, default=100,
        help='Number of nodes for example input (ONNX/TorchScript)'
    )
    parser.add_argument(
        '--num-edges', type=int, default=500,
        help='Number of edges for example input'
    )

    return parser.parse_args()


def export_pytorch(model, config, output_dir: Path):
    """Export model in PyTorch format."""
    print("\nExporting PyTorch format...")

    output_path = output_dir / 'graphsage_production.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'embedding_dim': model.out_channels,
        'in_channels': model.in_channels
    }, output_path)

    print(f"  Saved to: {output_path}")
    return output_path


def export_torchscript(model, num_nodes: int, num_edges: int, output_dir: Path):
    """Export model in TorchScript format."""
    print("\nExporting TorchScript format...")

    # Create example inputs
    example_features = torch.randn(num_nodes, model.in_channels)
    example_edges = torch.randint(0, num_nodes, (2, num_edges))

    # Script the model
    try:
        scripted_model = torch.jit.trace(model, (example_features, example_edges))
        output_path = output_dir / 'graphsage_scripted.pt'
        scripted_model.save(str(output_path))
        print(f"  Saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"  Warning: TorchScript export failed: {e}")
        return None


def export_onnx(model, num_nodes: int, num_edges: int, output_dir: Path):
    """Export model in ONNX format."""
    print("\nExporting ONNX format...")

    try:
        import onnx
    except ImportError:
        print("  Warning: onnx not installed. Install with: pip install onnx")
        return None

    # Create example inputs
    example_features = torch.randn(num_nodes, model.in_channels)
    example_edges = torch.randint(0, num_nodes, (2, num_edges))

    output_path = output_dir / 'graphsage.onnx'

    try:
        torch.onnx.export(
            model,
            (example_features, example_edges),
            str(output_path),
            input_names=['features', 'edge_index'],
            output_names=['embeddings'],
            dynamic_axes={
                'features': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'embeddings': {0: 'num_nodes'}
            },
            opset_version=14
        )
        print(f"  Saved to: {output_path}")

        # Verify
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX model verified successfully")

        return output_path
    except Exception as e:
        print(f"  Warning: ONNX export failed: {e}")
        return None


def main():
    """Main export function."""
    args = parse_args()

    print("=" * 60)
    print("GraphSAGE Model Export")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Get config
    config = checkpoint.get('config', load_config())
    model_config = config.get('model', {})
    feature_dim = config.get('features', {}).get('dimensions', 7)

    # Create model
    model = ProductionGraphSAGE(
        in_channels=feature_dim,
        hidden_channels=model_config.get('hidden_dim', 64),
        out_channels=model_config.get('output_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=0.0,  # No dropout for inference
        normalize_output=model_config.get('normalize_output', True)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: {model.count_parameters():,} parameters")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export based on format
    exported_paths = {}

    if args.format in ['pytorch', 'all']:
        path = export_pytorch(model, config, output_dir)
        exported_paths['pytorch'] = path

    if args.format in ['torchscript', 'all']:
        path = export_torchscript(model, args.num_nodes, args.num_edges, output_dir)
        if path:
            exported_paths['torchscript'] = path

    if args.format in ['onnx', 'all']:
        path = export_onnx(model, args.num_nodes, args.num_edges, output_dir)
        if path:
            exported_paths['onnx'] = path

    # Create metadata file
    metadata = {
        'source_checkpoint': args.checkpoint,
        'embedding_dim': model.out_channels,
        'in_channels': model.in_channels,
        'num_parameters': model.count_parameters(),
        'exported_formats': list(exported_paths.keys())
    }

    import json
    with open(output_dir / 'export_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Export Complete")
    print("=" * 60)
    print(f"\nExported files:")
    for fmt, path in exported_paths.items():
        if path:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {fmt}: {path} ({size_mb:.2f} MB)")

    print(f"\nMetadata: {output_dir / 'export_metadata.json'}")


if __name__ == '__main__':
    main()
