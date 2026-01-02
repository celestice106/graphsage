"""
Random Walk Generator Module.

This module generates random walks on the graph for skip-gram training.
Random walks capture local and global graph structure by sampling
paths through the graph.

Key Concept:
    Starting from each node, we perform multiple random walks. At each step,
    we randomly choose one of the current node's neighbors to visit next.
    This samples the graph structure in a way that can be used for training.
"""

import torch
import random
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np


class RandomWalkGenerator:
    """
    Generate random walks for skip-gram training.

    This class implements efficient random walk generation with:
    - Adjacency list for O(1) neighbor lookup
    - Handling of dead-ends (nodes with no outgoing edges)
    - Support for both directed and undirected walks
    - GPU-accelerated batch generation (optional)

    The generated walks are used to create (target, context) pairs
    for the skip-gram objective.

    Example:
        >>> import torch
        >>> from src.walks import RandomWalkGenerator
        >>>
        >>> # Create edge index (3 edges: 0->1, 0->2, 1->2)
        >>> edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
        >>> num_nodes = 3
        >>>
        >>> walker = RandomWalkGenerator(
        ...     edge_index=edge_index,
        ...     num_nodes=num_nodes,
        ...     walk_length=10,
        ...     walks_per_node=5
        ... )
        >>>
        >>> walks = walker.generate_all_walks()
        >>> print(f"Generated {len(walks)} walks")
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        walk_length: int = 80,
        walks_per_node: int = 10,
        seed: Optional[int] = 42
    ):
        """
        Initialize random walk generator.

        Args:
            edge_index: Edge tensor of shape [2, num_edges]
            num_nodes: Total number of nodes
            walk_length: Maximum length of each walk
            walks_per_node: Number of walks to start from each node
            seed: Random seed for reproducibility (None for random)
        """
        self.num_nodes = num_nodes
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build adjacency list for efficient neighbor access
        # adj_list[node] = list of neighbor nodes
        self.adj_list = self._build_adjacency_list(edge_index)

        # Pre-compute which nodes have neighbors (non-isolated)
        self.nodes_with_neighbors = [
            node for node in range(num_nodes)
            if len(self.adj_list[node]) > 0
        ]

    def _build_adjacency_list(self, edge_index: torch.Tensor) -> Dict[int, List[int]]:
        """
        Convert edge_index to adjacency list representation.

        The adjacency list provides O(1) access to neighbors, which is
        essential for efficient random walk generation.

        Args:
            edge_index: Edge tensor [2, num_edges]

        Returns:
            Dictionary mapping node -> list of neighbors
        """
        # Initialize empty neighbor lists for all nodes
        adj = {i: [] for i in range(self.num_nodes)}

        # Edge index might be on GPU, move to CPU for iteration
        if edge_index.numel() == 0:
            return adj

        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()

        for s, d in zip(src, dst):
            adj[s].append(d)

        return adj

    def generate_single_walk(self, start_node: int) -> List[int]:
        """
        Generate one random walk starting from start_node.

        The walk proceeds by randomly selecting a neighbor at each step.
        If a dead-end is reached (no neighbors), the walk terminates early.

        Args:
            start_node: Node to start the walk from

        Returns:
            List of node indices visited (including start)
        """
        walk = [start_node]
        current = start_node

        for _ in range(self.walk_length - 1):
            neighbors = self.adj_list[current]

            if not neighbors:
                # Dead end - stop walk
                break

            # Random neighbor selection
            current = random.choice(neighbors)
            walk.append(current)

        return walk

    def generate_walks_from_node(self, node: int) -> List[List[int]]:
        """
        Generate multiple walks from a single node.

        Args:
            node: Starting node

        Returns:
            List of walks (each walk is a list of node indices)
        """
        walks = []

        for _ in range(self.walks_per_node):
            walk = self.generate_single_walk(node)
            # Only keep walks with more than 1 node (actual movement)
            if len(walk) > 1:
                walks.append(walk)

        return walks

    def generate_all_walks(self, verbose: bool = False) -> List[List[int]]:
        """
        Generate walks starting from every node.

        This is the main method for generating training data.
        It creates walks_per_node walks from each node in the graph.

        Args:
            verbose: Whether to print progress

        Returns:
            List of all walks
        """
        all_walks = []

        for i, node in enumerate(range(self.num_nodes)):
            if verbose and i % 100 == 0:
                print(f"Generating walks: {i}/{self.num_nodes} nodes processed")

            walks = self.generate_walks_from_node(node)
            all_walks.extend(walks)

        if verbose:
            print(f"Generated {len(all_walks)} walks total")
            avg_length = np.mean([len(w) for w in all_walks]) if all_walks else 0
            print(f"Average walk length: {avg_length:.1f}")

        return all_walks

    def generate_walks_batch(
        self,
        nodes: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List[List[int]]:
        """
        Generate walks for a batch of nodes.

        Useful for generating walks incrementally or in parallel.

        Args:
            nodes: List of nodes to generate walks from. If None, uses all nodes.
            verbose: Whether to print progress

        Returns:
            List of walks
        """
        if nodes is None:
            nodes = list(range(self.num_nodes))

        all_walks = []

        for i, node in enumerate(nodes):
            if verbose and i % 100 == 0:
                print(f"Processing node {i}/{len(nodes)}")

            walks = self.generate_walks_from_node(node)
            all_walks.extend(walks)

        return all_walks

    def get_statistics(self) -> Dict:
        """
        Get statistics about the graph structure relevant to walking.

        Returns:
            Dictionary with graph walking statistics
        """
        degrees = [len(self.adj_list[n]) for n in range(self.num_nodes)]

        return {
            'num_nodes': self.num_nodes,
            'nodes_with_neighbors': len(self.nodes_with_neighbors),
            'isolated_nodes': self.num_nodes - len(self.nodes_with_neighbors),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'walk_length': self.walk_length,
            'walks_per_node': self.walks_per_node,
            'expected_total_walks': self.num_nodes * self.walks_per_node
        }


class BiasedRandomWalkGenerator(RandomWalkGenerator):
    """
    Random walk generator with node2vec-style biased sampling.

    This extends the basic random walk with parameters p and q that
    control the walk behavior:
    - p (return parameter): likelihood of returning to previous node
    - q (in-out parameter): likelihood of exploring outward vs staying local

    Note: This is more expensive than uniform random walks but can
    capture different structural properties.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        walk_length: int = 80,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        seed: Optional[int] = 42
    ):
        """
        Initialize biased random walk generator.

        Args:
            edge_index: Edge tensor of shape [2, num_edges]
            num_nodes: Total number of nodes
            walk_length: Maximum length of each walk
            walks_per_node: Number of walks per node
            p: Return parameter (1/p = probability of returning)
            q: In-out parameter (1/q = probability of going outward)
            seed: Random seed
        """
        super().__init__(edge_index, num_nodes, walk_length, walks_per_node, seed)

        self.p = p
        self.q = q

        # Build set version of neighbors for O(1) membership check
        self.neighbor_sets = {
            node: set(neighbors) for node, neighbors in self.adj_list.items()
        }

    def generate_single_walk(self, start_node: int) -> List[int]:
        """
        Generate one biased random walk.

        The bias affects neighbor selection based on:
        - Distance 0 (return to previous): weight = 1/p
        - Distance 1 (common neighbor): weight = 1
        - Distance 2 (exploration): weight = 1/q
        """
        walk = [start_node]
        current = start_node
        prev = None

        for _ in range(self.walk_length - 1):
            neighbors = self.adj_list[current]

            if not neighbors:
                break

            if prev is None:
                # First step: uniform random
                current = random.choice(neighbors)
            else:
                # Biased selection based on p and q
                weights = []
                prev_neighbors = self.neighbor_sets.get(prev, set())

                for neighbor in neighbors:
                    if neighbor == prev:
                        # Return to previous node
                        weights.append(1.0 / self.p)
                    elif neighbor in prev_neighbors:
                        # Common neighbor (distance 1 from prev)
                        weights.append(1.0)
                    else:
                        # Exploration (distance 2 from prev)
                        weights.append(1.0 / self.q)

                # Normalize and sample
                total = sum(weights)
                probs = [w / total for w in weights]
                idx = np.random.choice(len(neighbors), p=probs)
                prev = current
                current = neighbors[idx]

            walk.append(current)
            if prev is None:
                prev = start_node

        return walk
