"""
Custom Layer Wrappers for GraphSAGE.

This module provides utilities and wrapper classes for graph neural network
layers. Currently wraps PyTorch Geometric's SAGEConv with additional
functionality for our use case.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from typing import Optional


class SAGELayer(nn.Module):
    """
    Wrapped SAGEConv layer with optional batch normalization and activation.

    This provides a consistent interface for building GraphSAGE models with:
    - Configurable aggregation (mean, max)
    - Optional batch normalization
    - Optional activation function
    - Optional dropout

    Example:
        >>> layer = SAGELayer(
        ...     in_channels=7,
        ...     out_channels=64,
        ...     aggregator='mean',
        ...     use_bn=True,
        ...     activation='relu',
        ...     dropout=0.3
        ... )
        >>> out = layer(x, edge_index)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregator: str = 'mean',
        use_bn: bool = False,
        activation: Optional[str] = 'relu',
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Initialize SAGE layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            aggregator: Aggregation function ('mean' or 'max')
            use_bn: Whether to use batch normalization
            activation: Activation function ('relu', 'elu', None)
            dropout: Dropout rate
            bias: Whether to include bias
        """
        super().__init__()

        # Main convolution layer
        self.conv = SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggregator,
            bias=bias
        )

        # Optional batch normalization
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels)

        # Activation function
        self.activation = self._get_activation(activation)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _get_activation(self, name: Optional[str]) -> Optional[nn.Module]:
        """Get activation function by name."""
        if name is None:
            return None
        elif name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated features [num_nodes, out_channels]
        """
        # Message passing
        x = self.conv(x, edge_index)

        # Batch normalization
        if self.use_bn:
            x = self.bn(x)

        # Activation
        if self.activation is not None:
            x = self.activation(x)

        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.conv.reset_parameters()
        if self.use_bn:
            self.bn.reset_parameters()


class L2NormLayer(nn.Module):
    """
    L2 normalization layer.

    Normalizes embeddings to unit length, which is required for
    dot-product similarity in the skip-gram objective.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-12):
        """
        Initialize L2 normalization layer.

        Args:
            dim: Dimension along which to normalize
            eps: Small value for numerical stability
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to unit length.

        Args:
            x: Input tensor

        Returns:
            L2-normalized tensor
        """
        return torch.nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)


class MeanAggregator(nn.Module):
    """
    Mean aggregator for neighbor features.

    This is a simple aggregator that averages neighbor features.
    Used as a building block for custom GraphSAGE variants.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Aggregate neighbor features.

        Args:
            x: Node features [num_nodes, dim]
            edge_index: Edge indices [2, num_edges]
            size: (num_source_nodes, num_target_nodes) for bipartite graphs

        Returns:
            Aggregated features [num_nodes, dim]
        """
        from torch_geometric.utils import scatter

        row, col = edge_index
        num_nodes = size[1] if size is not None else x.size(0)

        # Sum features from neighbors
        out = scatter(x[row], col, dim=0, dim_size=num_nodes, reduce='mean')

        return out
