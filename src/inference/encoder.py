"""
Production Structural Encoder for Memory R1 Integration.

This module provides the main interface for computing structural embeddings
in the Memory R1 system. It handles:
- Loading trained GraphSAGE model
- Computing embeddings efficiently
- Caching for repeated access
- Integration with Memory R1's graph structure

The encoder is designed for sub-millisecond inference latency in production.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

from ..model.graphsage import ProductionGraphSAGE
from ..data.feature_extractor import MemoryFeatureExtractor
from ..data.view_extractor import GraphSAGEViewExtractor, GraphSAGEView
from .cache import EmbeddingCache


class MemoryR1StructuralEncoder:
    """
    Production structural encoder for Memory R1 integration.

    This is the main interface for computing structural embeddings:
    1. Extracts memory-only view from full graph
    2. Computes features for memory nodes
    3. Runs GraphSAGE forward pass
    4. Returns L2-normalized embeddings

    The encoder includes caching to avoid recomputation when the graph
    hasn't changed.

    Example:
        >>> from src.inference import MemoryR1StructuralEncoder
        >>>
        >>> # Initialize encoder
        >>> encoder = MemoryR1StructuralEncoder(
        ...     model_path='exports/graphsage_production.pt',
        ...     device='cuda'
        ... )
        >>>
        >>> # Get all embeddings
        >>> embeddings = encoder.encode_all(full_graph)
        >>>
        >>> # Get single embedding by original memory ID
        >>> emb = encoder.encode_by_id('mem_0001', full_graph)
        >>>
        >>> # Invalidate cache when graph changes
        >>> encoder.invalidate_cache()
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[ProductionGraphSAGE] = None,
        device: str = 'cuda',
        use_compile: bool = False,
        cache_embeddings: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Initialize structural encoder.

        Args:
            model_path: Path to saved model checkpoint
            model: Pre-loaded model (alternative to model_path)
            device: Device for inference ('cuda' or 'cpu')
            use_compile: Whether to use torch.compile for optimization
            cache_embeddings: Whether to cache computed embeddings
            config: Optional configuration dictionary
        """
        self.device = torch.device(device)
        self.config = config or {}

        # Load or use provided model
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Must provide either model_path or model")

        self.model.eval()

        # Optionally compile for faster inference
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, dynamic=False)
                print("Model compiled for optimized inference")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

        # Setup cache
        self.cache_embeddings = cache_embeddings
        self.cache = EmbeddingCache() if cache_embeddings else None

        # Tracking
        self.last_view: Optional[GraphSAGEView] = None
        self._inference_times: List[float] = []

    def _load_model(self, path: str) -> ProductionGraphSAGE:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Get model config from checkpoint or use defaults
        if 'config' in checkpoint:
            model_config = checkpoint['config'].get('model', {})
            feature_config = checkpoint['config'].get('features', {})
        else:
            model_config = self.config.get('model', {})
            feature_config = self.config.get('features', {})

        # Create model with saved config
        model = ProductionGraphSAGE(
            in_channels=feature_config.get('dimensions', 7),
            hidden_channels=model_config.get('hidden_dim', 64),
            out_channels=model_config.get('output_dim', 64),
            num_layers=model_config.get('num_layers', 2),
            dropout=0.0,  # No dropout at inference
            normalize_output=model_config.get('normalize_output', True)
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    @torch.no_grad()
    def encode_all(
        self,
        full_graph,
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Get embeddings for all memory nodes.

        Args:
            full_graph: Full Memory R1 graph (MockGraphStore or compatible)
            force_recompute: If True, ignore cache

        Returns:
            embeddings: Tensor [num_nodes, embedding_dim]
        """
        # Check cache first
        if self.cache is not None and not force_recompute:
            cached = self.cache.get_all()
            if cached is not None:
                return cached

        start_time = time.time()

        # Extract view from full graph
        view_extractor = GraphSAGEViewExtractor(full_graph)
        view = view_extractor.extract_with_undirected_edges()
        self.last_view = view

        # Compute features
        feature_extractor = MemoryFeatureExtractor(
            full_graph=full_graph,
            node_mapping=view.node_mapping,
            include_entity_features=True
        )
        features = feature_extractor.extract(view.edge_index)

        # Move to device
        features = features.to(self.device)
        edge_index = view.edge_index.to(self.device)

        # Forward pass
        embeddings = self.model(features, edge_index)

        # Update cache
        if self.cache is not None:
            self.cache.update(embeddings)

        # Track timing
        elapsed = time.time() - start_time
        self._inference_times.append(elapsed)

        return embeddings

    @torch.no_grad()
    def encode_single(
        self,
        node_idx: int,
        full_graph,
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Get embedding for single node by index.

        Note: This still computes all embeddings (GraphSAGE requires full graph),
        but returns only the requested one. Caching makes subsequent calls fast.

        Args:
            node_idx: Node index (0-indexed)
            full_graph: Full Memory R1 graph
            force_recompute: If True, ignore cache

        Returns:
            embedding: Tensor [embedding_dim]
        """
        embeddings = self.encode_all(full_graph, force_recompute)
        return embeddings[node_idx]

    @torch.no_grad()
    def encode_by_id(
        self,
        memory_id: str,
        full_graph,
        force_recompute: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Get embedding for node by original memory ID.

        Args:
            memory_id: Original memory node ID (e.g., 'mem_0001')
            full_graph: Full Memory R1 graph
            force_recompute: If True, ignore cache

        Returns:
            embedding: Tensor [embedding_dim] or None if ID not found
        """
        # Ensure we have the view
        if self.last_view is None or force_recompute:
            self.encode_all(full_graph, force_recompute)

        # Look up index
        if memory_id not in self.last_view.node_mapping:
            return None

        node_idx = self.last_view.node_mapping[memory_id]
        return self.cache.get_single(node_idx) if self.cache else None

    @torch.no_grad()
    def encode_batch(
        self,
        node_indices: List[int],
        full_graph,
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Get embeddings for batch of nodes.

        Args:
            node_indices: List of node indices
            full_graph: Full Memory R1 graph
            force_recompute: If True, ignore cache

        Returns:
            embeddings: Tensor [batch_size, embedding_dim]
        """
        all_embeddings = self.encode_all(full_graph, force_recompute)
        indices = torch.tensor(node_indices, device=self.device)
        return all_embeddings[indices]

    def invalidate_cache(self, node_ids: Optional[List[str]] = None):
        """
        Invalidate embedding cache.

        Call this when the graph structure changes (add/update/delete operations).

        Args:
            node_ids: Specific memory IDs to invalidate (None = all)
        """
        if self.cache is not None:
            if node_ids is None:
                self.cache.invalidate()
            else:
                # Convert memory IDs to indices
                if self.last_view is not None:
                    indices = [
                        self.last_view.node_mapping[mid]
                        for mid in node_ids
                        if mid in self.last_view.node_mapping
                    ]
                    self.cache.invalidate(indices)
                else:
                    self.cache.invalidate()

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.out_channels

    def get_statistics(self) -> Dict:
        """
        Get encoder statistics.

        Returns:
            Dictionary with inference stats
        """
        stats = {
            'embedding_dim': self.embedding_dim,
            'device': str(self.device),
            'cache_enabled': self.cache is not None,
        }

        if self._inference_times:
            times = self._inference_times[-100:]  # Last 100 inferences
            stats.update({
                'avg_inference_ms': sum(times) / len(times) * 1000,
                'min_inference_ms': min(times) * 1000,
                'max_inference_ms': max(times) * 1000,
                'num_inferences': len(self._inference_times)
            })

        if self.cache is not None:
            stats['cache'] = self.cache.get_statistics()

        return stats

    def save(self, path: str):
        """
        Save encoder for production deployment.

        Args:
            path: Output path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'embedding_dim': self.embedding_dim,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda', **kwargs) -> 'MemoryR1StructuralEncoder':
        """
        Load encoder from saved file.

        Args:
            path: Path to saved encoder
            device: Device for inference
            **kwargs: Additional arguments

        Returns:
            Loaded encoder
        """
        return cls(model_path=path, device=device, **kwargs)


def benchmark_encoder(
    encoder: MemoryR1StructuralEncoder,
    full_graph,
    num_iterations: int = 100
) -> Dict:
    """
    Benchmark encoder performance.

    Args:
        encoder: Encoder to benchmark
        full_graph: Graph to use for testing
        num_iterations: Number of iterations

    Returns:
        Benchmark results
    """
    import time

    # Warmup
    for _ in range(5):
        encoder.encode_all(full_graph, force_recompute=True)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        encoder.encode_all(full_graph, force_recompute=True)
        times.append(time.time() - start)

    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'throughput_per_sec': 1 / (sum(times) / len(times))
    }
