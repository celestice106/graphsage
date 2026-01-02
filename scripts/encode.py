#!/usr/bin/env python3
"""
GraphSAGE Structural Encoder - Simple Endpoint Script.

This script provides a simple API for encoding graphs into structural embeddings.
Supports multiple input formats: PyTorch Geometric, NetworkX, and raw edge lists.

Quick Start:
    >>> from scripts.encode import GraphSAGEEncoder
    >>>
    >>> # Load trained model
    >>> encoder = GraphSAGEEncoder('exports/graphsage_production.pt')
    >>>
    >>> # Encode a graph (auto-detects format)
    >>> embeddings = encoder.encode(graph)

Supported Formats:
    1. PyTorch Geometric Data object
    2. NetworkX Graph/DiGraph
    3. Dict with 'edge_index' and optional 'x' (features)
    4. Tuple of (edge_index, features)

Author: Memory R1 Team
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Dict, Tuple, List, Any
from pathlib import Path


class GraphSAGEEncoder:
    """
    Simple encoder for computing structural embeddings.

    This is the main user-facing API for the GraphSAGE structural encoder.
    It handles format conversion, feature computation, and model inference.

    Example:
        >>> encoder = GraphSAGEEncoder('model.pt', device='cuda')
        >>>
        >>> # From PyTorch Geometric
        >>> from torch_geometric.data import Data
        >>> data = Data(x=features, edge_index=edges)
        >>> embeddings = encoder.encode(data)
        >>>
        >>> # From NetworkX
        >>> import networkx as nx
        >>> G = nx.karate_club_graph()
        >>> embeddings = encoder.encode(G)
        >>>
        >>> # From edge list
        >>> edges = [[0, 1, 2], [1, 2, 0]]  # source, target pairs
        >>> embeddings = encoder.encode({'edge_index': edges})
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        auto_features: bool = True
    ):
        """
        Initialize the encoder.

        Args:
            model_path: Path to trained model (.pt file)
            device: Device for inference ('cuda', 'cpu', or 'auto')
            auto_features: If True, compute structural features automatically
                          when input graph has no node features
        """
        # Select device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.auto_features = auto_features
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"GraphSAGE Encoder loaded on {self.device}")
        print(f"  - Input dim: {self.input_dim}")
        print(f"  - Output dim: {self.output_dim}")

    def _load_model(self, path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Get config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})

        # Import model class
        from src.model.graphsage import ProductionGraphSAGE

        # Create model
        model = ProductionGraphSAGE(
            in_channels=config.get('features', {}).get('dimensions', 7),
            hidden_channels=model_config.get('hidden_dim', 64),
            out_channels=model_config.get('output_dim', 64),
            num_layers=model_config.get('num_layers', 2),
            dropout=0.0,  # No dropout at inference
            normalize_output=model_config.get('normalize_output', True)
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        # Store dimensions
        self._input_dim = model.in_channels
        self._output_dim = model.out_channels

        return model

    @property
    def input_dim(self) -> int:
        """Input feature dimension."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self._output_dim

    @torch.no_grad()
    def encode(
        self,
        graph: Any,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a graph into structural embeddings.

        This method auto-detects the input format and converts it appropriately.

        Args:
            graph: Input graph in any supported format:
                   - torch_geometric.data.Data
                   - networkx.Graph or networkx.DiGraph
                   - dict with 'edge_index' key
                   - tuple of (edge_index, features)
            features: Optional node features (overrides graph features)

        Returns:
            embeddings: Tensor of shape [num_nodes, output_dim], L2-normalized

        Raises:
            ValueError: If graph format is not recognized
        """
        # Convert to standard format
        edge_index, node_features, num_nodes = self._parse_input(graph, features)

        # Move to device
        edge_index = edge_index.to(self.device)
        node_features = node_features.to(self.device)

        # Forward pass
        embeddings = self.model(node_features, edge_index)

        return embeddings

    def _parse_input(
        self,
        graph: Any,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Parse input graph into standard format.

        Returns:
            Tuple of (edge_index, features, num_nodes)
        """
        # Check for PyTorch Geometric Data
        if self._is_pyg_data(graph):
            return self._from_pyg(graph, features)

        # Check for NetworkX graph
        if self._is_networkx(graph):
            return self._from_networkx(graph, features)

        # Check for dict format
        if isinstance(graph, dict):
            return self._from_dict(graph, features)

        # Check for tuple format
        if isinstance(graph, tuple) and len(graph) == 2:
            edge_index, feat = graph
            edge_index = self._to_tensor(edge_index)
            if features is None:
                features = feat
            features = self._to_tensor(features) if features is not None else None
            num_nodes = int(edge_index.max().item()) + 1
            if features is None:
                features = self._compute_structural_features(edge_index, num_nodes)
            return edge_index, features, num_nodes

        raise ValueError(
            f"Unrecognized graph format: {type(graph)}. "
            "Supported: PyG Data, NetworkX Graph, dict, tuple(edge_index, features)"
        )

    def _is_pyg_data(self, obj: Any) -> bool:
        """Check if object is PyG Data."""
        try:
            from torch_geometric.data import Data
            return isinstance(obj, Data)
        except ImportError:
            return False

    def _is_networkx(self, obj: Any) -> bool:
        """Check if object is NetworkX graph."""
        try:
            import networkx as nx
            return isinstance(obj, (nx.Graph, nx.DiGraph))
        except ImportError:
            return False

    def _from_pyg(
        self,
        data,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert PyG Data to standard format."""
        edge_index = data.edge_index
        num_nodes = data.num_nodes or int(edge_index.max().item()) + 1

        if features is not None:
            feat = features
        elif hasattr(data, 'x') and data.x is not None:
            feat = data.x
        else:
            feat = self._compute_structural_features(edge_index, num_nodes)

        return edge_index, feat, num_nodes

    def _from_networkx(
        self,
        G,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert NetworkX graph to standard format."""
        import networkx as nx

        # Get node mapping (ensure consistent ordering)
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        # Build edge index
        edges = list(G.edges())
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            src = [node_to_idx[e[0]] for e in edges]
            dst = [node_to_idx[e[1]] for e in edges]

            # Make undirected if it's a Graph (not DiGraph)
            if not isinstance(G, nx.DiGraph):
                src, dst = src + dst, dst + src

            edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Get or compute features
        if features is not None:
            feat = features
        elif 'x' in G.nodes[nodes[0]] if nodes else False:
            # Node features stored in graph
            feat = torch.tensor([G.nodes[n]['x'] for n in nodes], dtype=torch.float)
        else:
            feat = self._compute_structural_features(edge_index, num_nodes)

        return edge_index, feat, num_nodes

    def _from_dict(
        self,
        data: dict,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert dict format to standard format."""
        if 'edge_index' not in data:
            raise ValueError("Dict must contain 'edge_index' key")

        edge_index = self._to_tensor(data['edge_index'])
        num_nodes = data.get('num_nodes', int(edge_index.max().item()) + 1)

        if features is not None:
            feat = features
        elif 'x' in data and data['x'] is not None:
            feat = self._to_tensor(data['x'])
        elif 'features' in data and data['features'] is not None:
            feat = self._to_tensor(data['features'])
        else:
            feat = self._compute_structural_features(edge_index, num_nodes)

        return edge_index, feat, num_nodes

    def _to_tensor(self, x: Any) -> torch.Tensor:
        """Convert various types to tensor."""
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, (list, tuple)):
            return torch.tensor(x)
        raise ValueError(f"Cannot convert {type(x)} to tensor")

    def _compute_structural_features(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute structural features for nodes without features.

        Computes a 7-dimensional feature vector for each node based on
        graph structure (similar to Memory R1 features but generic).
        """
        if not self.auto_features:
            raise ValueError(
                "Graph has no node features and auto_features=False. "
                "Either provide features or set auto_features=True."
            )

        # Initialize features
        features = torch.zeros(num_nodes, 7, dtype=torch.float)

        if edge_index.numel() == 0:
            return features

        src, dst = edge_index[0], edge_index[1]

        # Feature 0: In-degree
        in_degree = torch.zeros(num_nodes)
        in_degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        features[:, 0] = in_degree

        # Feature 1: Out-degree
        out_degree = torch.zeros(num_nodes)
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        features[:, 1] = out_degree

        # Feature 2: Total degree
        features[:, 2] = in_degree + out_degree

        # Feature 3: Degree centrality (normalized)
        if num_nodes > 1:
            features[:, 3] = features[:, 2] / (num_nodes - 1)

        # Feature 4: Is source (has outgoing edges)
        features[:, 4] = (out_degree > 0).float()

        # Feature 5: Is sink (has incoming edges)
        features[:, 5] = (in_degree > 0).float()

        # Feature 6: Is isolated (no edges)
        features[:, 6] = ((in_degree == 0) & (out_degree == 0)).float()

        # Normalize features
        for i in range(4):  # Only normalize continuous features
            if features[:, i].max() > 0:
                features[:, i] = features[:, i] / features[:, i].max()

        return features

    def encode_networkx(
        self,
        G,
        node_features: Optional[Dict[Any, List[float]]] = None
    ) -> Dict[Any, np.ndarray]:
        """
        Encode NetworkX graph and return dict mapping node IDs to embeddings.

        This is a convenience method that preserves original node IDs.

        Args:
            G: NetworkX graph
            node_features: Optional dict mapping node IDs to feature vectors

        Returns:
            Dict mapping original node IDs to embedding arrays
        """
        import networkx as nx

        nodes = list(G.nodes())

        # Convert features if provided
        features = None
        if node_features is not None:
            features = torch.tensor(
                [node_features[n] for n in nodes],
                dtype=torch.float
            )

        # Encode
        embeddings = self.encode(G, features)

        # Map back to original IDs
        return {
            node: embeddings[i].cpu().numpy()
            for i, node in enumerate(nodes)
        }

    def similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings.

        Since embeddings are L2-normalized, this is just dot product.

        Args:
            emb1: First embedding(s) [dim] or [batch, dim]
            emb2: Second embedding(s) [dim] or [batch, dim]

        Returns:
            Similarity score(s)
        """
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)

        return (emb1 * emb2).sum(dim=-1)

    def most_similar(
        self,
        query_idx: int,
        embeddings: torch.Tensor,
        k: int = 5,
        exclude_self: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k most similar nodes to a query node.

        Args:
            query_idx: Index of query node
            embeddings: All node embeddings
            k: Number of similar nodes to return
            exclude_self: Whether to exclude the query node from results

        Returns:
            Tuple of (indices, similarities)
        """
        query = embeddings[query_idx]
        similarities = self.similarity(query, embeddings)

        if exclude_self:
            similarities[query_idx] = -float('inf')

        values, indices = torch.topk(similarities, k)
        return indices, values


def main():
    """Demo the encoder with a simple example."""
    import argparse

    parser = argparse.ArgumentParser(description='GraphSAGE Structural Encoder')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic graph')
    args = parser.parse_args()

    # Load encoder
    encoder = GraphSAGEEncoder(args.model)

    if args.demo:
        print("\n--- Demo: Encoding a synthetic graph ---\n")

        # Create a simple graph
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
        ], dtype=torch.long)

        graph = {'edge_index': edge_index, 'num_nodes': 5}

        # Encode
        embeddings = encoder.encode(graph)

        print(f"Input: 5-node ring graph")
        print(f"Output embeddings shape: {embeddings.shape}")
        print(f"Embedding norm (should be 1.0): {embeddings.norm(dim=1).mean():.4f}")

        # Find similar nodes
        print("\nMost similar nodes to node 0:")
        indices, sims = encoder.most_similar(0, embeddings, k=4)
        for idx, sim in zip(indices.tolist(), sims.tolist()):
            print(f"  Node {idx}: similarity = {sim:.4f}")


if __name__ == '__main__':
    main()
