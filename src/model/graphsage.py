"""
GraphSAGE Model.

This module implements the main GraphSAGE model for learning structural
embeddings of memory nodes. The model uses message passing to aggregate
neighbor features and produce embeddings that capture graph structure.

Architecture:
    Input Features [N, 7]
           │
           ▼
    SAGEConv(7, 64) + ReLU + Dropout
           │
           ▼
    SAGEConv(64, 64)
           │
           ▼
    L2 Normalization
           │
           ▼
    Output Embeddings [N, 64]

The L2 normalization is critical: it enables using dot product as
similarity metric, which is what the skip-gram loss requires.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional, List


class GraphSAGE(nn.Module):
    """
    GraphSAGE for Memory R1 structural embeddings.

    This model is designed for:
    - Quality embeddings that capture graph structure
    - Fast inference (single forward pass)
    - GPU training efficiency
    - Integration with Memory R1's dual embedding system

    The architecture follows the original GraphSAGE paper with:
    - Mean aggregation (most stable and effective)
    - 2 layers (captures 2-hop neighborhood)
    - L2 normalized outputs (for dot-product similarity)

    Example:
        >>> import torch
        >>> from src.model import GraphSAGE
        >>>
        >>> model = GraphSAGE(
        ...     in_channels=7,
        ...     hidden_channels=64,
        ...     out_channels=64,
        ...     dropout=0.3
        ... )
        >>> model = model.cuda()
        >>>
        >>> # Forward pass
        >>> features = torch.randn(100, 7).cuda()
        >>> edge_index = torch.randint(0, 100, (2, 500)).cuda()
        >>> embeddings = model(features, edge_index)
        >>> print(embeddings.shape)  # [100, 64]
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggregator: str = 'mean',
        normalize_output: bool = True
    ):
        """
        Initialize GraphSAGE model.

        Args:
            in_channels: Number of input features (7 for Memory R1)
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GraphSAGE layers (default 2)
            dropout: Dropout rate after first layer
            aggregator: Aggregation method ('mean', 'max', 'lstm')
            normalize_output: Whether to L2 normalize output embeddings
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.normalize_output = normalize_output

        # Build layers
        self.convs = nn.ModuleList()

        # First layer: in_channels -> hidden_channels
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))

        # Middle layers (if any): hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))

        # Last layer: hidden_channels -> out_channels
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute node embeddings.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            embeddings: L2-normalized embeddings [num_nodes, out_channels]
        """
        # Process through all layers except the last
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Last layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)

        # L2 normalize for dot-product similarity
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=1)

        return x

    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.out_channels

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FlexibleGraphSAGE(nn.Module):
    """
    Flexible GraphSAGE with configurable architecture.

    This version supports:
    - Variable number of layers
    - Different activation functions
    - Residual connections (optional)
    - Layer normalization (optional)

    Use this for experimentation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggregator: str = 'mean',
        activation: str = 'relu',
        use_layer_norm: bool = False,
        use_residual: bool = False,
        normalize_output: bool = True
    ):
        """
        Initialize flexible GraphSAGE.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of layers
            dropout: Dropout rate
            aggregator: Aggregation method
            activation: Activation function ('relu', 'elu', 'gelu')
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            normalize_output: Whether to L2 normalize outputs
        """
        super().__init__()

        self.num_layers = num_layers
        self.use_residual = use_residual
        self.normalize_output = normalize_output

        # Build convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))

        # Build layer norms if requested
        self.layer_norms = None
        if use_layer_norm:
            self.layer_norms = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # Residual projections (if dimensions don't match)
        self.residual_projs = None
        if use_residual:
            self.residual_projs = nn.ModuleList()
            # First layer residual
            if in_channels != hidden_channels:
                self.residual_projs.append(nn.Linear(in_channels, hidden_channels))
            else:
                self.residual_projs.append(nn.Identity())
            # Middle layers (identity if same dim)
            for _ in range(num_layers - 2):
                self.residual_projs.append(nn.Identity())
            # Last layer residual
            if num_layers > 1:
                if hidden_channels != out_channels:
                    self.residual_projs.append(nn.Linear(hidden_channels, out_channels))
                else:
                    self.residual_projs.append(nn.Identity())

        # Activation function
        self.activation = self._get_activation(activation)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            # Store input for residual
            if self.use_residual:
                residual = self.residual_projs[i](x)

            # Convolution
            x = conv(x, edge_index)

            # Skip activation and dropout for last layer
            if i < len(self.convs) - 1:
                # Layer norm
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)

                # Activation
                x = self.activation(x)

                # Residual
                if self.use_residual:
                    x = x + residual

                # Dropout
                x = self.dropout(x)
            else:
                # Last layer: just residual if enabled
                if self.use_residual:
                    x = x + residual

        # L2 normalize
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=1)

        return x


def create_model(config: dict, device: Optional[torch.device] = None) -> GraphSAGE:
    """
    Create GraphSAGE model from config.

    Args:
        config: Configuration dictionary with model parameters
        device: Device to place model on

    Returns:
        GraphSAGE model
    """
    model_config = config.get('model', {})

    model = GraphSAGE(
        in_channels=config.get('features', {}).get('dimensions', 7),
        hidden_channels=model_config.get('hidden_dim', 64),
        out_channels=model_config.get('output_dim', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.3),
        aggregator=model_config.get('aggregator', 'mean'),
        normalize_output=model_config.get('normalize_output', True)
    )

    if device is not None:
        model = model.to(device)

    return model
