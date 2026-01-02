"""
GraphSAGE View Extractor Module.

This module extracts a memory-only view from the full Memory R1 heterogeneous
graph. The view is used for GraphSAGE training while the full graph remains
unchanged.

Key Concept:
    The full Memory R1 graph contains:
    - Memory nodes (facts, events, experiences)
    - Entity nodes (semantic anchors)
    - 3 edge types (caused_by, next_event, mention)

    GraphSAGE operates on a simplified view containing:
    - Only Memory nodes
    - Only memory-to-memory edges (caused_by, next_event)
    - Entity information encoded as node features
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .graph_loader import MockGraphStore, EdgeType


@dataclass
class GraphSAGEView:
    """
    Container for GraphSAGE-compatible graph view.

    Attributes:
        memory_ids: List of original memory node IDs (preserves order)
        edge_index: PyTorch tensor of shape [2, num_edges] with node indices
        node_mapping: Dict mapping original memory ID to 0-indexed position
        reverse_mapping: Dict mapping index back to original memory ID
        num_nodes: Total number of memory nodes
        num_edges: Total number of edges
    """
    memory_ids: List[str]
    edge_index: torch.Tensor
    node_mapping: Dict[str, int]
    reverse_mapping: Dict[int, str]
    num_nodes: int
    num_edges: int

    def to(self, device: torch.device) -> 'GraphSAGEView':
        """Move edge_index to specified device."""
        return GraphSAGEView(
            memory_ids=self.memory_ids,
            edge_index=self.edge_index.to(device),
            node_mapping=self.node_mapping,
            reverse_mapping=self.reverse_mapping,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges
        )


class GraphSAGEViewExtractor:
    """
    Extract memory-only view from full Memory R1 graph.

    This class creates a simplified graph view suitable for GraphSAGE training:
    1. Filters to keep only Memory nodes (Entity nodes excluded)
    2. Keeps only memory-to-memory edges (caused_by, next_event)
    3. Re-indexes nodes from 0 to N-1 for PyTorch compatibility
    4. Creates bidirectional mapping between original IDs and indices

    The full graph remains completely unchanged - this is a READ-ONLY operation.

    Example:
        >>> from src.data import MockGraphStore, GraphSAGEViewExtractor
        >>>
        >>> # Create or load full graph
        >>> store = MockGraphStore()
        >>> store.generate_synthetic_graph(num_memories=500)
        >>>
        >>> # Extract view for GraphSAGE
        >>> extractor = GraphSAGEViewExtractor(store)
        >>> view = extractor.extract()
        >>>
        >>> print(f"Memory nodes: {view.num_nodes}")
        >>> print(f"Edges: {view.num_edges}")
        >>> print(f"Edge index shape: {view.edge_index.shape}")
    """

    def __init__(self, full_graph: MockGraphStore):
        """
        Initialize view extractor.

        Args:
            full_graph: Full Memory R1 graph (MockGraphStore or compatible)
        """
        self.full_graph = full_graph

    def extract(self, include_edge_types: Optional[List[str]] = None) -> GraphSAGEView:
        """
        Extract GraphSAGE-compatible view from full graph.

        This method:
        1. Gets all memory nodes from the graph
        2. Creates a mapping from original IDs to sequential indices
        3. Filters edges to only include memory-to-memory connections
        4. Converts edges to PyTorch tensor format

        Args:
            include_edge_types: Edge types to include. Defaults to
                               ["caused_by", "next_event"]

        Returns:
            GraphSAGEView containing the extracted subgraph
        """
        if include_edge_types is None:
            include_edge_types = ["caused_by", "next_event"]

        # Step 1: Get all memory nodes
        memory_nodes = self.full_graph.get_nodes_by_type("memory")
        memory_ids = [node.id for node in memory_nodes]

        # Step 2: Create bidirectional mapping (original_id <-> index)
        # Sorting ensures deterministic ordering across runs
        memory_ids = sorted(memory_ids)
        node_mapping = {mid: idx for idx, mid in enumerate(memory_ids)}
        reverse_mapping = {idx: mid for mid, idx in node_mapping.items()}

        # Step 3: Get memory-to-memory edges only
        edges = []
        for edge_type in include_edge_types:
            edge_list = self.full_graph.get_edges_by_type([edge_type])
            for src, dst, _ in edge_list:
                # Only include edges where BOTH endpoints are memory nodes
                if src in node_mapping and dst in node_mapping:
                    # Convert to indices
                    src_idx = node_mapping[src]
                    dst_idx = node_mapping[dst]
                    edges.append([src_idx, dst_idx])

        # Step 4: Convert to PyTorch tensor
        if edges:
            # edge_index shape: [2, num_edges]
            # Row 0: source indices
            # Row 1: target indices
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Empty graph - create empty tensor with correct shape
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return GraphSAGEView(
            memory_ids=memory_ids,
            edge_index=edge_index,
            node_mapping=node_mapping,
            reverse_mapping=reverse_mapping,
            num_nodes=len(memory_ids),
            num_edges=edge_index.shape[1]
        )

    def extract_with_undirected_edges(
        self,
        include_edge_types: Optional[List[str]] = None
    ) -> GraphSAGEView:
        """
        Extract view with edges treated as undirected.

        For GraphSAGE, message passing often benefits from undirected edges
        (information flows both ways). This method adds reverse edges for
        each directed edge in the original graph.

        Note: This doubles the edge count but improves message passing.

        Args:
            include_edge_types: Edge types to include

        Returns:
            GraphSAGEView with undirected edges
        """
        # First extract directed view
        view = self.extract(include_edge_types)

        if view.num_edges > 0:
            # Add reverse edges: if (u, v) exists, add (v, u)
            forward_edges = view.edge_index
            reverse_edges = view.edge_index.flip(0)  # Swap rows

            # Concatenate and remove duplicates
            all_edges = torch.cat([forward_edges, reverse_edges], dim=1)

            # Remove duplicate edges (optional, but cleaner)
            all_edges = torch.unique(all_edges, dim=1)

            return GraphSAGEView(
                memory_ids=view.memory_ids,
                edge_index=all_edges,
                node_mapping=view.node_mapping,
                reverse_mapping=view.reverse_mapping,
                num_nodes=view.num_nodes,
                num_edges=all_edges.shape[1]
            )
        else:
            return view

    def get_statistics(self) -> Dict:
        """
        Get statistics about the extracted view.

        Returns:
            Dictionary with graph statistics
        """
        view = self.extract()

        # Compute degree statistics
        if view.num_edges > 0:
            # Out-degree (edges leaving each node)
            src_nodes = view.edge_index[0]
            out_degrees = torch.bincount(src_nodes, minlength=view.num_nodes)

            # In-degree (edges entering each node)
            dst_nodes = view.edge_index[1]
            in_degrees = torch.bincount(dst_nodes, minlength=view.num_nodes)

            total_degrees = out_degrees + in_degrees
        else:
            out_degrees = torch.zeros(view.num_nodes, dtype=torch.long)
            in_degrees = torch.zeros(view.num_nodes, dtype=torch.long)
            total_degrees = torch.zeros(view.num_nodes, dtype=torch.long)

        return {
            'num_nodes': view.num_nodes,
            'num_edges': view.num_edges,
            'avg_degree': total_degrees.float().mean().item() if view.num_nodes > 0 else 0,
            'max_degree': total_degrees.max().item() if view.num_nodes > 0 else 0,
            'min_degree': total_degrees.min().item() if view.num_nodes > 0 else 0,
            'isolated_nodes': (total_degrees == 0).sum().item(),
            'density': view.num_edges / (view.num_nodes * (view.num_nodes - 1))
                       if view.num_nodes > 1 else 0
        }


def extract_graphsage_view(full_graph: MockGraphStore, undirected: bool = True) -> GraphSAGEView:
    """
    Convenience function to extract GraphSAGE view.

    Args:
        full_graph: Full Memory R1 graph
        undirected: Whether to treat edges as undirected

    Returns:
        GraphSAGEView ready for training
    """
    extractor = GraphSAGEViewExtractor(full_graph)

    if undirected:
        return extractor.extract_with_undirected_edges()
    else:
        return extractor.extract()
