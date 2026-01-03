"""
Graph Loader Module.

This module provides utilities for loading graphs from Memory Bank system
and creating mock graphs for testing/development.

The GraphLoader class serves as an adapter between Memory Bank's graph storage
and the GraphSAGE training pipeline.
"""

import json
import torch
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """Types of nodes in Memory Bank graph."""
    MEMORY = "memory"
    ENTITY = "entity"


class EdgeType(Enum):
    """Types of edges in Memory Bank graph."""
    CAUSED_BY = "caused_by"
    NEXT_EVENT = "next_event"
    MENTION = "mention"


@dataclass
class MemoryNode:
    """
    Represents a memory node in the graph.

    Attributes:
        id: Unique identifier for the memory
        content: The text content of the memory
        timestamp: When the memory was created
        session_id: Which session this memory belongs to
        status: Current status (active, deleted, etc.)
    """
    id: str
    content: str = ""
    timestamp: float = 0.0
    session_id: str = ""
    status: str = "active"


@dataclass
class EntityNode:
    """
    Represents an entity node in the graph.

    Attributes:
        id: Unique identifier for the entity
        name: Entity name
        entity_type: Type of entity (person, location, etc.)
        aliases: Alternative names for this entity
    """
    id: str
    name: str = ""
    entity_type: str = "unknown"
    aliases: List[str] = field(default_factory=list)


@dataclass
class Edge:
    """
    Represents an edge in the graph.

    Attributes:
        source: Source node ID
        target: Target node ID
        edge_type: Type of relationship
        properties: Additional edge properties
    """
    source: str
    target: str
    edge_type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)


class MockGraphStore:
    """
    Mock graph store for testing and development.

    This class simulates the Memory Bank graph store interface,
    allowing GraphSAGE training without requiring the full Memory Bank system.

    The mock store creates a realistic heterogeneous graph with:
    - Memory nodes (facts, events, experiences)
    - Entity nodes (semantic anchors)
    - Three edge types (caused_by, next_event, mention)

    Example:
        >>> store = MockGraphStore()
        >>> store.generate_synthetic_graph(num_memories=100, num_entities=30)
        >>> memories = store.get_nodes_by_type("memory")
        >>> edges = store.get_edges_by_type(["caused_by", "next_event"])
    """

    def __init__(self):
        """Initialize empty mock graph store."""
        # Internal storage using NetworkX for convenience
        self.graph = nx.DiGraph()

        # Separate indices for fast lookup
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.entity_nodes: Dict[str, EntityNode] = {}
        self.edges_by_type: Dict[EdgeType, List[Edge]] = {
            EdgeType.CAUSED_BY: [],
            EdgeType.NEXT_EVENT: [],
            EdgeType.MENTION: [],
        }

        # Entity to memories mapping (for shared entity features)
        self.entity_to_memories: Dict[str, Set[str]] = {}

    def add_memory_node(self, node: MemoryNode) -> None:
        """Add a memory node to the graph."""
        self.memory_nodes[node.id] = node
        self.graph.add_node(node.id, node_type=NodeType.MEMORY.value, **vars(node))

    def add_entity_node(self, node: EntityNode) -> None:
        """Add an entity node to the graph."""
        self.entity_nodes[node.id] = node
        self.graph.add_node(node.id, node_type=NodeType.ENTITY.value, **vars(node))
        if node.id not in self.entity_to_memories:
            self.entity_to_memories[node.id] = set()

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges_by_type[edge.edge_type].append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            edge_type=edge.edge_type.value,
            **edge.properties
        )

        # Track entity-memory associations
        if edge.edge_type == EdgeType.MENTION:
            entity_id = edge.target
            memory_id = edge.source
            if entity_id in self.entity_to_memories:
                self.entity_to_memories[entity_id].add(memory_id)

    def get_nodes_by_type(self, node_type: str) -> List[Any]:
        """
        Get all nodes of a specific type.

        Args:
            node_type: "memory" or "entity"

        Returns:
            List of node objects
        """
        if node_type == "memory":
            return list(self.memory_nodes.values())
        elif node_type == "entity":
            return list(self.entity_nodes.values())
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def get_edges_by_type(self, edge_types: List[str]) -> List[Tuple[str, str, Dict]]:
        """
        Get all edges of specified types.

        Args:
            edge_types: List of edge type strings

        Returns:
            List of (source, target, properties) tuples
        """
        result = []
        for edge_type_str in edge_types:
            edge_type = EdgeType(edge_type_str)
            for edge in self.edges_by_type[edge_type]:
                result.append((edge.source, edge.target, edge.properties))
        return result

    def get_mentioned_entities(self, memory_id: str) -> List[str]:
        """
        Get all entities mentioned by a memory.

        Args:
            memory_id: ID of the memory node

        Returns:
            List of entity IDs
        """
        entities = []
        for edge in self.edges_by_type[EdgeType.MENTION]:
            if edge.source == memory_id:
                entities.append(edge.target)
        return entities

    def get_memories_mentioning(self, entity_id: str) -> List[str]:
        """
        Get all memories that mention an entity.

        Args:
            entity_id: ID of the entity node

        Returns:
            List of memory IDs
        """
        return list(self.entity_to_memories.get(entity_id, set()))

    def generate_synthetic_graph(
        self,
        num_memories: int = 200,
        num_entities: int = 50,
        causal_density: float = 0.05,
        temporal_density: float = 0.08,
        entity_mention_rate: float = 0.3,
        seed: int = 42
    ) -> None:
        """
        Generate a synthetic graph for testing.

        Creates a realistic Memory Bank graph structure with:
        - Memory nodes with varied connectivity
        - Entity nodes as semantic anchors
        - Causal chains (caused_by edges)
        - Temporal sequences (next_event edges)
        - Entity mentions (mention edges)

        Args:
            num_memories: Number of memory nodes to create
            num_entities: Number of entity nodes to create
            causal_density: Probability of causal edge between memories
            temporal_density: Probability of temporal edge between memories
            entity_mention_rate: Average entities mentioned per memory
            seed: Random seed for reproducibility
        """
        import random
        random.seed(seed)

        # Clear existing data
        self.graph.clear()
        self.memory_nodes.clear()
        self.entity_nodes.clear()
        for edge_type in self.edges_by_type:
            self.edges_by_type[edge_type] = []
        self.entity_to_memories.clear()

        # --- Create entity nodes ---
        # Entities represent people, places, concepts that memories reference
        entity_types = ["person", "location", "organization", "concept", "object"]
        for i in range(num_entities):
            entity = EntityNode(
                id=f"entity_{i:04d}",
                name=f"Entity {i}",
                entity_type=random.choice(entity_types),
                aliases=[f"entity{i}", f"e{i}"]
            )
            self.add_entity_node(entity)

        entity_ids = list(self.entity_nodes.keys())

        # --- Create memory nodes ---
        # Memories represent facts, events, experiences
        sessions = [f"session_{j}" for j in range(max(1, num_memories // 20))]
        for i in range(num_memories):
            memory = MemoryNode(
                id=f"mem_{i:04d}",
                content=f"Memory content {i}",
                timestamp=float(i),  # Sequential timestamps
                session_id=random.choice(sessions),
                status="active"
            )
            self.add_memory_node(memory)

        memory_ids = list(self.memory_nodes.keys())

        # --- Create caused_by edges ---
        # Causal relationships: "X happened because of Y"
        # These form directed acyclic subgraphs (effects point to causes)
        for i, src_id in enumerate(memory_ids):
            # Earlier memories can cause later ones (ensures DAG structure)
            for j in range(i + 1, min(i + 20, num_memories)):  # Look ahead window
                if random.random() < causal_density:
                    dst_id = memory_ids[j]
                    edge = Edge(
                        source=dst_id,  # Effect points to cause
                        target=src_id,
                        edge_type=EdgeType.CAUSED_BY
                    )
                    self.add_edge(edge)

        # --- Create next_event edges ---
        # Temporal relationships: "X happened, then Y happened"
        # These form temporal chains within sessions
        session_memories: Dict[str, List[str]] = {}
        for mem_id, mem in self.memory_nodes.items():
            session = mem.session_id
            if session not in session_memories:
                session_memories[session] = []
            session_memories[session].append(mem_id)

        for session, mems in session_memories.items():
            # Sort by timestamp (which is just the index for synthetic data)
            mems_sorted = sorted(mems, key=lambda x: self.memory_nodes[x].timestamp)
            for i in range(len(mems_sorted) - 1):
                if random.random() < temporal_density * 3:  # Higher rate within sessions
                    edge = Edge(
                        source=mems_sorted[i],
                        target=mems_sorted[i + 1],
                        edge_type=EdgeType.NEXT_EVENT
                    )
                    self.add_edge(edge)

        # --- Create mention edges ---
        # Entity mentions: memories reference entities
        # Use power-law distribution for entity popularity
        entity_weights = [1.0 / (i + 1) ** 0.5 for i in range(num_entities)]
        weight_sum = sum(entity_weights)
        entity_probs = [w / weight_sum for w in entity_weights]

        for mem_id in memory_ids:
            # Poisson-like number of entities per memory
            num_mentions = max(0, int(random.gauss(entity_mention_rate * num_entities, 2)))
            num_mentions = min(num_mentions, num_entities // 3)

            if num_mentions > 0:
                # Sample entities with popularity bias
                mentioned = set()
                for _ in range(num_mentions):
                    idx = random.choices(range(num_entities), weights=entity_probs)[0]
                    mentioned.add(entity_ids[idx])

                for entity_id in mentioned:
                    edge = Edge(
                        source=mem_id,
                        target=entity_id,
                        edge_type=EdgeType.MENTION
                    )
                    self.add_edge(edge)

        # Log statistics
        print(f"Generated synthetic graph:")
        print(f"  Memory nodes: {len(self.memory_nodes)}")
        print(f"  Entity nodes: {len(self.entity_nodes)}")
        print(f"  caused_by edges: {len(self.edges_by_type[EdgeType.CAUSED_BY])}")
        print(f"  next_event edges: {len(self.edges_by_type[EdgeType.NEXT_EVENT])}")
        print(f"  mention edges: {len(self.edges_by_type[EdgeType.MENTION])}")

    def save(self, path: str) -> None:
        """
        Save graph to JSON file.

        Args:
            path: Output file path
        """
        data = {
            'memory_nodes': [
                {'id': n.id, 'content': n.content, 'timestamp': n.timestamp,
                 'session_id': n.session_id, 'status': n.status}
                for n in self.memory_nodes.values()
            ],
            'entity_nodes': [
                {'id': n.id, 'name': n.name, 'entity_type': n.entity_type,
                 'aliases': n.aliases}
                for n in self.entity_nodes.values()
            ],
            'edges': [
                {'source': e.source, 'target': e.target,
                 'edge_type': e.edge_type.value, 'properties': e.properties}
                for edges in self.edges_by_type.values()
                for e in edges
            ]
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load graph from JSON file.

        Args:
            path: Input file path
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Clear existing data
        self.graph.clear()
        self.memory_nodes.clear()
        self.entity_nodes.clear()
        for edge_type in self.edges_by_type:
            self.edges_by_type[edge_type] = []
        self.entity_to_memories.clear()

        # Load memory nodes
        for node_data in data['memory_nodes']:
            node = MemoryNode(**node_data)
            self.add_memory_node(node)

        # Load entity nodes
        for node_data in data['entity_nodes']:
            node = EntityNode(**node_data)
            self.add_entity_node(node)

        # Load edges
        for edge_data in data['edges']:
            edge = Edge(
                source=edge_data['source'],
                target=edge_data['target'],
                edge_type=EdgeType(edge_data['edge_type']),
                properties=edge_data.get('properties', {})
            )
            self.add_edge(edge)


class GraphLoader:
    """
    Load graphs from various sources for GraphSAGE training.

    This class provides a unified interface for loading graphs from:
    - Memory Bank graph store 
    - JSON files (saved graphs)
    - Mock graph store (testing)

    Example:
        >>> loader = GraphLoader()
        >>>
        >>> # Load from Memory Bank (when integrated)
        >>> graph = loader.from_memory_bank(memory_bank)
        >>>
        >>> # Load from file
        >>> graph = loader.from_file("data/raw/graph.json")
        >>>
        >>> # Create mock graph
        >>> graph = loader.create_mock(num_memories=500)
    """

    @staticmethod
    def from_memory_bank(memory_bank: Any) -> MockGraphStore:
        """
        Load graph from Memory Bank.

        This method extracts the graph structure from a Memory Bank
        and wraps it in a MockGraphStore for uniform interface.

        Args:
            memory_bank: Memory Bank instance

        Returns:
            MockGraphStore populated with Memory Bank data
        """
        store = MockGraphStore()

        # Get graph store from memory bank
        # Assumes memory_bank.graph_store is a NetworkX-compatible graph
        if hasattr(memory_bank, 'graph_store'):
            nx_graph = memory_bank.graph_store.graph

            # Extract memory nodes
            for node_id in nx_graph.nodes():
                node_data = nx_graph.nodes[node_id]
                if node_data.get('node_type') == 'memory':
                    memory = MemoryNode(
                        id=node_id,
                        content=node_data.get('content', ''),
                        timestamp=node_data.get('timestamp', 0.0),
                        session_id=node_data.get('session_id', ''),
                        status=node_data.get('status', 'active')
                    )
                    store.add_memory_node(memory)
                elif node_data.get('node_type') == 'entity':
                    entity = EntityNode(
                        id=node_id,
                        name=node_data.get('name', ''),
                        entity_type=node_data.get('entity_type', 'unknown'),
                        aliases=node_data.get('aliases', [])
                    )
                    store.add_entity_node(entity)

            # Extract edges
            for src, dst, edge_data in nx_graph.edges(data=True):
                edge_type_str = edge_data.get('edge_type', 'caused_by')
                try:
                    edge_type = EdgeType(edge_type_str)
                except ValueError:
                    continue  # Skip unknown edge types

                edge = Edge(
                    source=src,
                    target=dst,
                    edge_type=edge_type,
                    properties={k: v for k, v in edge_data.items() if k != 'edge_type'}
                )
                store.add_edge(edge)

        return store

    @staticmethod
    def from_file(path: str) -> MockGraphStore:
        """
        Load graph from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            MockGraphStore populated from file
        """
        store = MockGraphStore()
        store.load(path)
        return store

    @staticmethod
    def create_mock(
        num_memories: int = 200,
        num_entities: int = 50,
        seed: int = 42,
        **kwargs
    ) -> MockGraphStore:
        """
        Create a mock graph for testing.

        Args:
            num_memories: Number of memory nodes
            num_entities: Number of entity nodes
            seed: Random seed
            **kwargs: Additional parameters for generate_synthetic_graph

        Returns:
            MockGraphStore with synthetic data
        """
        store = MockGraphStore()
        store.generate_synthetic_graph(
            num_memories=num_memories,
            num_entities=num_entities,
            seed=seed,
            **kwargs
        )
        return store
