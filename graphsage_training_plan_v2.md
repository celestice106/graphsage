# GraphSAGE Training Plan for Memory R1

**GDS-Style Random Walk Co-occurrence Approach**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Approach](#solution-approach)
3. [Graph Architecture](#graph-architecture)
4. [Project Structure](#project-structure)
5. [Implementation Pipeline](#implementation-pipeline)
6. [Execution Plan](#execution-plan)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Memory R1 Integration](#memory-r1-integration)
9. [Timeline and Milestones](#timeline-and-milestones)

---

## Problem Statement

### Background

The Memory R1 system requires a dual embedding architecture combining text embeddings (semantic) with structural embeddings (topological) for effective memory retrieval and management. The structural embeddings must capture how memory nodes relate to each other in the graph topology.

### Failed Approaches

Several approaches to training GraphSAGE for structural embeddings have been attempted and failed:

| Approach | Feature Dimensions | Failure Mode |
|----------|-------------------|--------------|
| Degree + LDP | 7 + 6 = 13 dims | Model converged to 50/50 random guessing on link prediction |
| RWPE (Random Walk Positional Encoding) | 13 dims | Return probabilities collapsed to zero after step 3-4 due to graph sparsity |
| Degree + LDP + Spectral | 20 dims | Training stuck, no convergence |

### Root Cause Analysis

The failures stem from two fundamental issues:

**Issue 1: Insufficient Training Signal (Link Prediction)**
- Small graph (~hundreds of nodes from 152 QA pairs) provides limited positive edges
- Class imbalance: O(n) edges vs O(n²) non-edges overwhelms the model
- Not enough examples of "what a connection looks like" for the model to learn

**Issue 2: Feature Collapse (RWPE)**
- Return probability requires paths back to the starting node
- Sparse graphs have few such paths
- After 3-4 steps, return probability approaches zero for all nodes
- Features become identical (all zeros), providing no discriminative signal

### Requirements

The structural embedding solution must:

- Work on sparse, evolving graphs (memory nodes added/deleted during RL training)
- Provide embeddings every RL step for the Memory Management Agent (MMA)
- Achieve sub-millisecond inference latency for production use
- Capture meaningful topological relationships without requiring dense connectivity
- Integrate seamlessly with the existing Memory R1 dual embedding architecture
- Handle heterogeneous graph structure (Memory + Entity nodes, 3 edge types)

---

## Solution Approach

### Core Idea: Random Walk Co-occurrence

Instead of predicting edge existence (link prediction) or return probabilities (RWPE), we adopt the **random walk co-occurrence** approach used by Neo4j Graph Data Science (GDS) library.

**Key Insight**: We don't ask "will node u return to itself?" — we ask "which nodes appear together when walking the graph?"

### How It Works

```
1. Generate random walks starting from each node
2. Nodes appearing together within a context window are "similar"
3. Train GraphSAGE to produce embeddings where co-occurring nodes have high dot-product similarity
4. Use skip-gram objective (same as Word2Vec, but for graphs)
```

### Why This Solves Our Problems

| Problem | Link Prediction | RWPE | Random Walk Co-occurrence |
|---------|-----------------|------|---------------------------|
| Sparse graph | Few positive edges | Return prob → 0 | Walks still traverse available edges |
| Training signal | O(edges) pairs | O(nodes) features | O(walks × length × nodes) pairs |
| Feature collapse | N/A | All zeros | N/A (no return prob needed) |
| Small graph | Insufficient data | Insufficient paths | Amplifies signal via multiple walks |

### Training Signal Amplification

With default parameters on a 500-node graph:

```
walks_per_node = 10
walk_length = 80
context_window = 10
nodes = 500

Positive pairs ≈ 500 × 10 × 80 × 10 = 4,000,000 training pairs
```

Compare to link prediction: ~1,000-2,000 edges = 1,000-2,000 positive pairs.

**4000x more training signal** from the same graph.

---

## Graph Architecture

### Full Graph Design (Unchanged)

Your Memory R1 system uses a heterogeneous graph that **remains exactly as designed**:

```
┌─────────────────────────────────────────────────────────────────┐
│                  MEMORY R1 FULL GRAPH                           │
│                  (Your Original Design)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   NODE TYPES:                                                   │
│   ┌──────────────┐         ┌──────────────┐                    │
│   │ Memory Node  │         │ Entity Node  │                    │
│   │              │         │              │                    │
│   │ • content    │         │ • name       │                    │
│   │ • timestamp  │         │ • type       │                    │
│   │ • session_id │         │ • aliases    │                    │
│   │ • status     │         │              │                    │
│   └──────────────┘         └──────────────┘                    │
│                                                                 │
│   EDGE TYPES:                                                   │
│   Memory ───caused_by───► Memory    (causal chains)            │
│   Memory ───next_event──► Memory    (temporal sequences)       │
│   Memory ───mention─────► Entity    (entity grounding)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Design Is Correct

| Component | Purpose | Critical For |
|-----------|---------|--------------|
| **Memory nodes** | Store facts, events, experiences | Core retrieval, RL state |
| **Entity nodes** | Semantic anchors, deduplication | Entity grounding, disambiguation |
| **caused_by** | Causal reasoning chains | "Why did X happen?" queries |
| **next_event** | Temporal sequences | "What happened after X?" queries |
| **mention** | Link memories to entities | Entity-centric retrieval |

### GraphSAGE View (Derived from Full Graph)

GraphSAGE operates on a **view** of the full graph, not a replacement:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Full Graph (Your Design) ✓ UNCHANGED          │   │
│  │                                                         │   │
│  │   Memory ──caused_by──► Memory                          │   │
│  │   Memory ──next_event──► Memory                         │   │
│  │   Memory ──mention──► Entity                            │   │
│  │                                                         │   │
│  │   Used for: Storage, Queries, Traversal, Deduplication │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           │ extract_graphsage_view()            │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           GraphSAGE View (Computed at Runtime)          │   │
│  │                                                         │   │
│  │   Nodes: Memory nodes only                              │   │
│  │   Edges: caused_by + next_event                         │   │
│  │   Features: degree + entity-derived features            │   │
│  │                                                         │   │
│  │   Used for: Structural embedding computation only       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### View Extraction Logic

```python
def extract_graphsage_view(full_graph):
    """
    Extract memory-only subgraph for GraphSAGE.
    Full graph remains unchanged.
    """
    # 1. Get Memory nodes only
    memory_nodes = full_graph.get_nodes_by_type("memory")
    memory_node_ids = [n.id for n in memory_nodes]
    
    # 2. Get memory-to-memory edges only
    edges = full_graph.get_edges_by_type(["caused_by", "next_event"])
    edge_index = filter_edges_to_nodes(edges, memory_node_ids)
    
    # 3. Compute features (including entity-derived)
    features = compute_memory_features(memory_nodes, full_graph)
    
    return memory_node_ids, edge_index, features
```

### Feature Design for Memory Nodes

Since Entity information is valuable but Entity nodes aren't in the GraphSAGE view, we encode entity information as **features** of Memory nodes:

| Feature | Dimension | Source | Rationale |
|---------|-----------|--------|-----------|
| caused_by degree (in+out) | 1 | caused_by edges | Causal connectivity |
| next_event degree (in+out) | 1 | next_event edges | Temporal connectivity |
| num_entities_mentioned | 1 | mention edges | Entity richness |
| shared_entity_neighbors | 1 | mention edges | Co-reference potential |
| is_cause (has outgoing caused_by) | 1 | caused_by edges | Structural role |
| is_effect (has incoming caused_by) | 1 | caused_by edges | Structural role |
| has_successor (has outgoing next_event) | 1 | next_event edges | Temporal position |
| **Total** | **7** | | |

---

## Project Structure

```
memory_r1_graphsage/
│
├── README.md                       # This planning document
├── requirements.txt                # Dependencies
├── setup.py                        # Package installation
│
├── config/
│   ├── __init__.py
│   ├── default.yaml               # Default hyperparameters
│   └── production.yaml            # Production settings
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                      # Data generation & processing
│   │   ├── __init__.py
│   │   ├── graph_loader.py       # Load graph from Memory R1
│   │   ├── view_extractor.py     # Extract GraphSAGE view from full graph
│   │   ├── feature_extractor.py  # Compute node features
│   │   └── dataset.py            # PyTorch Dataset wrapper
│   │
│   ├── walks/                     # Random walk generation
│   │   ├── __init__.py
│   │   ├── generator.py          # RandomWalkGenerator class
│   │   ├── pair_sampler.py       # Co-occurrence pair extraction
│   │   └── negative_sampler.py   # Degree-biased negative sampling
│   │
│   ├── model/                     # GraphSAGE model
│   │   ├── __init__.py
│   │   ├── layers.py             # SAGEConv layers wrapper
│   │   ├── graphsage.py          # ProductionGraphSAGE model
│   │   └── loss.py               # Skip-gram loss function
│   │
│   ├── training/                  # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py            # GraphSAGETrainer class
│   │   ├── batch_generator.py    # Training batch generation
│   │   └── callbacks.py          # Logging, checkpoints, early stopping
│   │
│   ├── inference/                 # Production inference
│   │   ├── __init__.py
│   │   ├── encoder.py            # MemoryR1StructuralEncoder
│   │   └── cache.py              # Embedding cache with dirty tracking
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── graph_utils.py        # Graph manipulation helpers
│       ├── metrics.py            # Evaluation metrics
│       └── visualization.py      # Embedding visualization
│
├── scripts/
│   ├── generate_data.py           # Generate training data from graph
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Evaluation script
│   ├── export_model.py            # Export for production
│   └── benchmark_latency.py       # Latency benchmarking
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py               # Data generation tests
│   ├── test_walks.py              # Random walk tests
│   ├── test_model.py              # Model forward pass tests
│   ├── test_training.py           # Training loop tests
│   └── test_integration.py        # End-to-end integration tests
│
├── notebooks/
│   ├── 01_graph_exploration.ipynb # Explore Memory R1 graph structure
│   ├── 02_feature_analysis.ipynb  # Analyze feature distributions
│   ├── 03_walk_analysis.ipynb     # Visualize random walks
│   ├── 04_training_debug.ipynb    # Debug training issues
│   └── 05_embedding_quality.ipynb # Visualize and evaluate embeddings
│
├── data/                          # Generated data artifacts
│   ├── raw/                       # Raw graph exports
│   ├── processed/                 # Processed views and features
│   └── walks/                     # Generated walk data
│
├── checkpoints/                   # Saved model checkpoints
│   └── .gitkeep
│
├── logs/                          # Training logs
│   └── .gitkeep
│
└── exports/                       # Production model exports
    └── .gitkeep
```

---

## Implementation Pipeline

The implementation follows a clear **4-stage pipeline**: Data Generation → Data Processing → Training → Inference.

```
┌─────────────────────────────────────────────────────────────────┐
│                   IMPLEMENTATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    STAGE 1   │    │    STAGE 2   │    │    STAGE 3   │      │
│  │     Data     │───►│     Data     │───►│   Training   │      │
│  │  Generation  │    │  Processing  │    │    System    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ • Graph view │    │ • Features   │    │ • Model      │      │
│  │ • Node list  │    │ • Walk pairs │    │ • Optimizer  │      │
│  │ • Edge index │    │ • Negatives  │    │ • Loss       │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                 │               │
│                                                 ▼               │
│                                          ┌──────────────┐      │
│                                          │    STAGE 4   │      │
│                                          │  Inference   │      │
│                                          │   System     │      │
│                                          └──────────────┘      │
│                                                 │               │
│                                                 ▼               │
│                                          ┌──────────────┐      │
│                                          │ • Encoder    │      │
│                                          │ • Cache      │      │
│                                          │ • Integration│      │
│                                          └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Data Generation

**Objective**: Extract GraphSAGE view from Memory R1 full graph.

**Input**: Memory R1 full graph (Memory nodes, Entity nodes, 3 edge types)

**Output**: GraphSAGE-compatible graph (Memory nodes only, 2 edge types)

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: DATA GENERATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1.1: Load Full Graph                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input: Memory R1 GraphStore                            │   │
│  │  Output: NetworkX graph or edge list                    │   │
│  │  File: src/data/graph_loader.py                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 1.2: Extract Memory-Only View                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input: Full heterogeneous graph                        │   │
│  │  Process:                                               │   │
│  │    • Filter nodes: keep only type="memory"              │   │
│  │    • Filter edges: keep only caused_by, next_event      │   │
│  │    • Reindex nodes: 0 to N-1                            │   │
│  │    • Build edge_index tensor: [2, num_edges]            │   │
│  │  Output: memory_ids, edge_index, node_id_mapping        │   │
│  │  File: src/data/view_extractor.py                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 1.3: Save Raw Data                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output files:                                          │   │
│  │    • data/raw/memory_nodes.json                         │   │
│  │    • data/raw/edge_index.pt                             │   │
│  │    • data/raw/node_mapping.json                         │   │
│  │    • data/raw/full_graph_snapshot.json (for features)   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Code: View Extractor**

```python
# src/data/view_extractor.py

class GraphSAGEViewExtractor:
    """Extract memory-only view from full Memory R1 graph."""
    
    def __init__(self, full_graph):
        self.full_graph = full_graph
    
    def extract(self):
        """
        Extract GraphSAGE-compatible view.
        
        Returns:
            memory_ids: List of original memory node IDs
            edge_index: torch.Tensor [2, num_edges]
            node_mapping: Dict mapping original_id -> new_index
        """
        # Step 1: Get memory nodes
        memory_nodes = self.full_graph.get_nodes_by_type("memory")
        memory_ids = [n.id for n in memory_nodes]
        
        # Step 2: Create mapping (original_id -> 0-indexed)
        node_mapping = {mid: idx for idx, mid in enumerate(memory_ids)}
        
        # Step 3: Get memory-to-memory edges
        edges = []
        for edge_type in ["caused_by", "next_event"]:
            for src, dst, _ in self.full_graph.get_edges_by_type(edge_type):
                if src in node_mapping and dst in node_mapping:
                    edges.append([node_mapping[src], node_mapping[dst]])
        
        # Step 4: Convert to tensor
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return memory_ids, edge_index, node_mapping
```

---

### Stage 2: Data Processing

**Objective**: Compute features, generate walks, extract training pairs.

**Input**: GraphSAGE view (memory nodes, edge_index)

**Output**: Training-ready data (features, positive pairs, negative samples)

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: DATA PROCESSING                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 2.1: Feature Extraction                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input: memory_ids, edge_index, full_graph              │   │
│  │  Process:                                               │   │
│  │    For each memory node:                                │   │
│  │      • Count caused_by edges (in + out)                 │   │
│  │      • Count next_event edges (in + out)                │   │
│  │      • Count mentioned entities                         │   │
│  │      • Count memories sharing entities                  │   │
│  │      • Compute binary role indicators                   │   │
│  │    Normalize all features                               │   │
│  │  Output: features tensor [num_nodes, 7]                 │   │
│  │  File: src/data/feature_extractor.py                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 2.2: Random Walk Generation                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input: edge_index, num_nodes                           │   │
│  │  Parameters:                                            │   │
│  │    • walk_length: 80                                    │   │
│  │    • walks_per_node: 10                                 │   │
│  │  Process:                                               │   │
│  │    • Build adjacency list from edge_index               │   │
│  │    • For each node, generate walks_per_node walks       │   │
│  │    • Handle dead-ends (stop walk if no neighbors)       │   │
│  │  Output: walks list [num_nodes * walks_per_node, <=80]  │   │
│  │  File: src/walks/generator.py                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 2.3: Positive Pair Extraction                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input: walks list                                      │   │
│  │  Parameters:                                            │   │
│  │    • context_window: 10                                 │   │
│  │  Process:                                               │   │
│  │    For each walk:                                       │   │
│  │      For each position i:                               │   │
│  │        target = walk[i]                                 │   │
│  │        contexts = walk[i-window : i+window+1]           │   │
│  │        Add pairs (target, context) for each context     │   │
│  │  Output: positive_pairs list [(target, context), ...]   │   │
│  │  File: src/walks/pair_sampler.py                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 2.4: Negative Sampler Setup                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input: edge_index, num_nodes                           │   │
│  │  Parameters:                                            │   │
│  │    • num_negatives: 5                                   │   │
│  │    • distribution: degree^0.75                          │   │
│  │  Process:                                               │   │
│  │    • Compute degree for each node                       │   │
│  │    • Compute sampling probability: P(v) ∝ deg(v)^0.75   │   │
│  │    • Create multinomial sampler                         │   │
│  │  Output: NegativeSampler object                         │   │
│  │  File: src/walks/negative_sampler.py                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 2.5: Save Processed Data                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output files:                                          │   │
│  │    • data/processed/features.pt                         │   │
│  │    • data/processed/edge_index.pt                       │   │
│  │    • data/walks/positive_pairs.pt                       │   │
│  │    • data/processed/degree_distribution.pt              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Code: Feature Extractor**

```python
# src/data/feature_extractor.py

class MemoryFeatureExtractor:
    """Compute features for memory nodes including entity-derived features."""
    
    def __init__(self, full_graph, node_mapping):
        self.full_graph = full_graph
        self.node_mapping = node_mapping  # original_id -> index
        self.num_nodes = len(node_mapping)
    
    def extract(self, edge_index):
        """
        Compute 7-dimensional features for each memory node.
        
        Returns:
            features: torch.Tensor [num_nodes, 7]
        """
        features = torch.zeros(self.num_nodes, 7)
        
        # Compute edge-type-specific degrees
        caused_by_deg = self._compute_typed_degree("caused_by")
        next_event_deg = self._compute_typed_degree("next_event")
        
        for original_id, idx in self.node_mapping.items():
            # Feature 0: caused_by degree (normalized)
            features[idx, 0] = caused_by_deg.get(original_id, 0)
            
            # Feature 1: next_event degree (normalized)
            features[idx, 1] = next_event_deg.get(original_id, 0)
            
            # Feature 2: num entities mentioned
            entities = self.full_graph.get_mentioned_entities(original_id)
            features[idx, 2] = len(entities)
            
            # Feature 3: shared entity neighbors
            features[idx, 3] = self._count_shared_entity_neighbors(original_id)
            
            # Feature 4: is_cause (has outgoing caused_by)
            features[idx, 4] = float(self._has_outgoing("caused_by", original_id))
            
            # Feature 5: is_effect (has incoming caused_by)
            features[idx, 5] = float(self._has_incoming("caused_by", original_id))
            
            # Feature 6: has_successor (has outgoing next_event)
            features[idx, 6] = float(self._has_outgoing("next_event", original_id))
        
        # Normalize continuous features (columns 0-3)
        features[:, :4] = self._log_normalize(features[:, :4])
        
        return features
    
    def _log_normalize(self, x):
        """Log-normalize to prevent large value dominance."""
        return torch.log1p(x) / (torch.log1p(x).max(dim=0).values + 1e-8)
    
    def _count_shared_entity_neighbors(self, memory_id):
        """Count memories that share at least one entity with this memory."""
        my_entities = set(self.full_graph.get_mentioned_entities(memory_id))
        if not my_entities:
            return 0
        
        neighbors = set()
        for entity in my_entities:
            mentioning = self.full_graph.get_memories_mentioning(entity)
            neighbors.update(mentioning)
        
        neighbors.discard(memory_id)  # Remove self
        return len(neighbors)
```

**Key Code: Random Walk Generator**

```python
# src/walks/generator.py

class RandomWalkGenerator:
    """Generate random walks for skip-gram training."""
    
    def __init__(self, edge_index, num_nodes, walk_length=80, walks_per_node=10):
        self.num_nodes = num_nodes
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        
        # Build adjacency list for O(1) neighbor access
        self.adj_list = self._build_adjacency_list(edge_index)
    
    def _build_adjacency_list(self, edge_index):
        """Convert edge_index to adjacency list."""
        adj = {i: [] for i in range(self.num_nodes)}
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)
        return adj
    
    def generate_single_walk(self, start_node):
        """Generate one random walk from start_node."""
        walk = [start_node]
        current = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = self.adj_list[current]
            if not neighbors:
                break  # Dead end
            current = random.choice(neighbors)
            walk.append(current)
        
        return walk
    
    def generate_all_walks(self):
        """Generate walks_per_node walks for each node."""
        all_walks = []
        
        for node in range(self.num_nodes):
            for _ in range(self.walks_per_node):
                walk = self.generate_single_walk(node)
                if len(walk) > 1:  # Only keep non-trivial walks
                    all_walks.append(walk)
        
        return all_walks
```

**Key Code: Pair Sampler**

```python
# src/walks/pair_sampler.py

class CooccurrencePairSampler:
    """Extract (target, context) pairs from walks."""
    
    def __init__(self, context_window=10):
        self.context_window = context_window
    
    def extract_pairs(self, walks):
        """
        Extract all positive pairs from walks.
        
        Returns:
            pairs: List of (target, context) tuples
        """
        pairs = []
        
        for walk in walks:
            walk_len = len(walk)
            
            for i, target in enumerate(walk):
                # Define context window boundaries
                start = max(0, i - self.context_window)
                end = min(walk_len, i + self.context_window + 1)
                
                # Add pairs for all context nodes
                for j in range(start, end):
                    if i != j:
                        pairs.append((target, walk[j]))
        
        return pairs
```

**Key Code: Negative Sampler**

```python
# src/walks/negative_sampler.py

class DegreeBiasedNegativeSampler:
    """Degree-biased negative sampling with sublinear dampening."""
    
    def __init__(self, edge_index, num_nodes, exponent=0.75):
        self.num_nodes = num_nodes
        
        # Compute degree
        src = edge_index[0]
        degree = torch.zeros(num_nodes)
        for node in src.tolist():
            degree[node] += 1
        
        # Apply sublinear dampening: P(v) ∝ deg(v)^0.75
        self.probs = (degree ** exponent)
        self.probs = self.probs / self.probs.sum()
    
    def sample(self, num_samples, num_negatives=5):
        """
        Sample negative nodes.
        
        Args:
            num_samples: Number of positive pairs (batch size)
            num_negatives: Negatives per positive
            
        Returns:
            negatives: torch.Tensor [num_samples, num_negatives]
        """
        total = num_samples * num_negatives
        samples = torch.multinomial(self.probs, total, replacement=True)
        return samples.view(num_samples, num_negatives)
```

---

### Stage 3: Training System

**Objective**: Train GraphSAGE with skip-gram objective.

**Input**: Features, edge_index, positive pairs, negative sampler

**Output**: Trained GraphSAGE model

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 3: TRAINING SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 3.1: Model Definition                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Architecture:                                          │   │
│  │    Input [N, 7]                                         │   │
│  │      ↓                                                  │   │
│  │    SAGEConv(7, 64) + ReLU + Dropout(0.3)               │   │
│  │      ↓                                                  │   │
│  │    SAGEConv(64, 64)                                    │   │
│  │      ↓                                                  │   │
│  │    L2 Normalize                                        │   │
│  │      ↓                                                  │   │
│  │    Output [N, 64]                                       │   │
│  │  File: src/model/graphsage.py                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 3.2: Loss Function                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Skip-gram with negative sampling:                      │   │
│  │                                                         │   │
│  │  L = -log(σ(e_target · e_context))                     │   │
│  │      - Σ log(σ(-e_target · e_negative))                │   │
│  │                                                         │   │
│  │  Where σ is sigmoid, · is dot product                  │   │
│  │  File: src/model/loss.py                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 3.3: Batch Generator                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input: positive_pairs, negative_sampler                │   │
│  │  Process:                                               │   │
│  │    • Shuffle positive pairs                             │   │
│  │    • Chunk into batches of size 512                     │   │
│  │    • For each batch, sample negatives on-the-fly        │   │
│  │  Output: Iterator of (targets, contexts, negatives)     │   │
│  │  File: src/training/batch_generator.py                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 3.4: Training Loop                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  For each epoch:                                        │   │
│  │    For each batch (targets, contexts, negatives):       │   │
│  │      1. Forward pass: emb = model(features, edge_index) │   │
│  │      2. Compute positive scores: target · context       │   │
│  │      3. Compute negative scores: target · negatives     │   │
│  │      4. Compute skip-gram loss                          │   │
│  │      5. Backward pass + optimizer step                  │   │
│  │    Log epoch loss                                       │   │
│  │    Save checkpoint if best                              │   │
│  │    Early stopping if plateau                            │   │
│  │  File: src/training/trainer.py                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 3.5: Save Trained Model                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output files:                                          │   │
│  │    • checkpoints/best_model.pt (state_dict)             │   │
│  │    • checkpoints/training_config.yaml                   │   │
│  │    • logs/training_metrics.json                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Code: GraphSAGE Model**

```python
# src/model/graphsage.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class ProductionGraphSAGE(nn.Module):
    """Production-ready 2-layer GraphSAGE for Memory R1."""
    
    def __init__(self, in_channels=7, hidden_channels=64, out_channels=64, dropout=0.3):
        super().__init__()
        
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            embeddings: L2-normalized embeddings [num_nodes, out_channels]
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        # L2 normalize for dot-product similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x
```

**Key Code: Skip-gram Loss**

```python
# src/model/loss.py

import torch
import torch.nn.functional as F

class SkipGramLoss(nn.Module):
    """Skip-gram loss with negative sampling."""
    
    def forward(self, embeddings, targets, contexts, negatives):
        """
        Compute skip-gram loss.
        
        Args:
            embeddings: All node embeddings [num_nodes, dim]
            targets: Target node indices [batch_size]
            contexts: Context node indices [batch_size]
            negatives: Negative node indices [batch_size, num_negatives]
            
        Returns:
            loss: Scalar loss value
        """
        # Get embeddings for each role
        target_emb = embeddings[targets]      # [batch, dim]
        context_emb = embeddings[contexts]    # [batch, dim]
        negative_emb = embeddings[negatives]  # [batch, num_neg, dim]
        
        # Positive scores: dot product of (target, context)
        pos_scores = (target_emb * context_emb).sum(dim=1)  # [batch]
        
        # Negative scores: dot product of (target, each negative)
        # target_emb: [batch, dim] -> [batch, dim, 1]
        # negative_emb: [batch, num_neg, dim]
        # bmm result: [batch, num_neg, 1] -> [batch, num_neg]
        neg_scores = torch.bmm(
            negative_emb,
            target_emb.unsqueeze(2)
        ).squeeze(2)  # [batch, num_neg]
        
        # Skip-gram loss
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        
        return pos_loss + neg_loss
```

**Key Code: Trainer**

```python
# src/training/trainer.py

class GraphSAGETrainer:
    """Complete training pipeline."""
    
    def __init__(self, model, features, edge_index, positive_pairs, 
                 negative_sampler, config, device='cpu'):
        self.model = model.to(device)
        self.features = features.to(device)
        self.edge_index = edge_index.to(device)
        self.positive_pairs = positive_pairs
        self.negative_sampler = negative_sampler
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate']
        )
        self.loss_fn = SkipGramLoss()
        
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle pairs
        pairs = self.positive_pairs.copy()
        random.shuffle(pairs)
        
        # Process batches
        batch_size = self.config['batch_size']
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            
            # Extract targets and contexts
            targets = torch.tensor([p[0] for p in batch_pairs], device=self.device)
            contexts = torch.tensor([p[1] for p in batch_pairs], device=self.device)
            
            # Sample negatives
            negatives = self.negative_sampler.sample(
                len(batch_pairs), 
                self.config['num_negatives']
            ).to(self.device)
            
            # Forward pass (full batch)
            self.optimizer.zero_grad()
            embeddings = self.model(self.features, self.edge_index)
            
            # Compute loss
            loss = self.loss_fn(embeddings, targets, contexts, negatives)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip']
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs):
        """Full training loop with early stopping."""
        for epoch in range(num_epochs):
            loss = self.train_epoch()
            
            # Logging
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
            
            # Checkpointing
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint('checkpoints/best_model.pt')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return self.best_loss
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }, path)
```

---

### Stage 4: Inference System

**Objective**: Production-ready encoder for Memory R1 integration.

**Input**: Trained model, graph data

**Output**: Fast embeddings with caching

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 4: INFERENCE SYSTEM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 4.1: Production Encoder                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Features:                                              │   │
│  │    • Load trained model                                 │   │
│  │    • Apply torch.compile for 3-5x speedup              │   │
│  │    • Provide encode_all() and encode_single() methods   │   │
│  │  File: src/inference/encoder.py                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 4.2: Embedding Cache                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Features:                                              │   │
│  │    • Cache computed embeddings                          │   │
│  │    • Track dirty nodes (graph changed)                  │   │
│  │    • Invalidate on add/update/delete operations         │   │
│  │    • Lazy recomputation on access                       │   │
│  │  File: src/inference/cache.py                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 4.3: Model Export                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Formats:                                               │   │
│  │    • TorchScript: torch.jit.script(model)              │   │
│  │    • ONNX: torch.onnx.export(model, ...)               │   │
│  │  File: scripts/export_model.py                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Step 4.4: Benchmark                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Metrics:                                               │   │
│  │    • Inference latency (target: <1ms)                   │   │
│  │    • Memory footprint                                   │   │
│  │    • Throughput (embeddings/second)                     │   │
│  │  File: scripts/benchmark_latency.py                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Code: Production Encoder**

```python
# src/inference/encoder.py

class MemoryR1StructuralEncoder:
    """Production structural encoder for Memory R1 integration."""
    
    def __init__(self, model_path, feature_extractor, device='cpu'):
        self.device = device
        self.feature_extractor = feature_extractor
        
        # Load model
        self.model = ProductionGraphSAGE()
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Compile for production
        self.model = torch.compile(self.model, dynamic=False)
        
        # Embedding cache
        self.cache = EmbeddingCache()
    
    @torch.no_grad()
    def encode_all(self, edge_index, num_nodes, full_graph):
        """
        Get embeddings for all memory nodes.
        
        Returns:
            embeddings: torch.Tensor [num_nodes, 64]
        """
        # Check cache
        if self.cache.is_valid():
            return self.cache.get_all()
        
        # Compute features
        features = self.feature_extractor.extract(edge_index, full_graph)
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Forward pass
        embeddings = self.model(features, edge_index)
        
        # Update cache
        self.cache.update(embeddings)
        
        return embeddings
    
    @torch.no_grad()
    def encode_single(self, node_idx, edge_index, num_nodes, full_graph):
        """
        Get embedding for single node.
        
        Returns:
            embedding: torch.Tensor [64]
        """
        embeddings = self.encode_all(edge_index, num_nodes, full_graph)
        return embeddings[node_idx]
    
    def invalidate_cache(self, node_ids=None):
        """Invalidate cache when graph changes."""
        self.cache.invalidate(node_ids)
    
    @property
    def embedding_dim(self):
        return 64
```

**Key Code: Embedding Cache**

```python
# src/inference/cache.py

class EmbeddingCache:
    """Cache embeddings with dirty tracking."""
    
    def __init__(self):
        self.embeddings = None
        self.valid = False
        self.dirty_nodes = set()
    
    def is_valid(self):
        """Check if cache is valid (no dirty nodes)."""
        return self.valid and len(self.dirty_nodes) == 0
    
    def get_all(self):
        """Get cached embeddings."""
        return self.embeddings
    
    def update(self, embeddings):
        """Update cache with new embeddings."""
        self.embeddings = embeddings
        self.valid = True
        self.dirty_nodes.clear()
    
    def invalidate(self, node_ids=None):
        """
        Invalidate cache.
        
        Args:
            node_ids: Specific nodes to invalidate, or None for all
        """
        if node_ids is None:
            self.valid = False
            self.embeddings = None
        else:
            self.dirty_nodes.update(node_ids)
```

---

## Execution Plan

### Phase 1: Foundation (Days 1-2)

| Task | Description | Output |
|------|-------------|--------|
| 1.1 | Create project directory structure | All folders and `__init__.py` files |
| 1.2 | Define configuration schema | `config/default.yaml` |
| 1.3 | Implement graph loader | `src/data/graph_loader.py` |
| 1.4 | Implement view extractor | `src/data/view_extractor.py` |
| 1.5 | Write data generation tests | `tests/test_data.py` |

**Checkpoint**: Can extract memory-only view from full graph.

### Phase 2: Data Processing (Days 3-5)

| Task | Description | Output |
|------|-------------|--------|
| 2.1 | Implement feature extractor | `src/data/feature_extractor.py` |
| 2.2 | Implement walk generator | `src/walks/generator.py` |
| 2.3 | Implement pair sampler | `src/walks/pair_sampler.py` |
| 2.4 | Implement negative sampler | `src/walks/negative_sampler.py` |
| 2.5 | Create data generation script | `scripts/generate_data.py` |
| 2.6 | Write processing tests | `tests/test_walks.py` |

**Checkpoint**: Can generate millions of training pairs from graph.

### Phase 3: Training System (Days 6-9)

| Task | Description | Output |
|------|-------------|--------|
| 3.1 | Implement GraphSAGE model | `src/model/graphsage.py` |
| 3.2 | Implement skip-gram loss | `src/model/loss.py` |
| 3.3 | Implement batch generator | `src/training/batch_generator.py` |
| 3.4 | Implement trainer | `src/training/trainer.py` |
| 3.5 | Implement callbacks | `src/training/callbacks.py` |
| 3.6 | Create training script | `scripts/train.py` |
| 3.7 | Write training tests | `tests/test_training.py` |

**Checkpoint**: Training converges, loss < 1.0.

### Phase 4: Evaluation (Days 10-11)

| Task | Description | Output |
|------|-------------|--------|
| 4.1 | Implement evaluation metrics | `src/utils/metrics.py` |
| 4.2 | Implement visualization | `src/utils/visualization.py` |
| 4.3 | Create evaluation script | `scripts/evaluate.py` |
| 4.4 | Run embedding analysis | `notebooks/05_embedding_quality.ipynb` |

**Checkpoint**: Embeddings show meaningful structure.

### Phase 5: Production & Integration (Days 12-14)

| Task | Description | Output |
|------|-------------|--------|
| 5.1 | Implement production encoder | `src/inference/encoder.py` |
| 5.2 | Implement embedding cache | `src/inference/cache.py` |
| 5.3 | Create export script | `scripts/export_model.py` |
| 5.4 | Create benchmark script | `scripts/benchmark_latency.py` |
| 5.5 | Integrate with Memory R1 | Update Memory R1 files |
| 5.6 | Write integration tests | `tests/test_integration.py` |

**Checkpoint**: <1ms inference, Memory R1 integration complete.

---

## Evaluation Strategy

### Quantitative Metrics

| Category | Metric | Target |
|----------|--------|--------|
| **Training** | Final loss | < 1.0 |
| **Training** | Loss decrease | Monotonic first 20 epochs |
| **Embedding** | Neighbor similarity | > 0.5 |
| **Embedding** | Link prediction AUC | > 0.75 |
| **Production** | Inference latency | < 5ms (target < 1ms) |
| **Production** | Memory footprint | < 50MB |

### Failure Mode Detection

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss stuck at ~1.4 | No learning | Check gradients, increase LR |
| All embeddings identical | Feature collapse | Verify feature variance |
| Loss oscillating | LR too high | Reduce LR, add gradient clipping |
| Embeddings uncorrelated with structure | Wrong objective | Verify walk generation |

---

## Memory R1 Integration

### Integration Points

```python
# memory_r1_bank/embeddings/structural_encoder.py

from memory_r1_graphsage.src.inference.encoder import MemoryR1StructuralEncoder

class StructuralEmbeddingModule:
    """Wrapper for GraphSAGE in Memory R1."""
    
    def __init__(self, model_path='exports/graphsage_production.pt'):
        self.encoder = MemoryR1StructuralEncoder(model_path)
    
    def get_embedding(self, memory_id, graph_store):
        """Get structural embedding for memory node."""
        view = extract_graphsage_view(graph_store)
        node_idx = view.node_mapping[memory_id]
        return self.encoder.encode_single(node_idx, view.edge_index, ...)
    
    def on_graph_change(self, changed_memory_ids):
        """Called when memory operations modify graph."""
        self.encoder.invalidate_cache(changed_memory_ids)
```

### MMA State Construction

```python
# memory_r1_bank/interfaces/mma_interface.py

def get_mma_state(self, input_text, input_entities):
    """Construct RL state for Memory Management Agent."""
    
    # Text embedding (existing)
    text_emb = self.text_encoder.encode(input_text)  # [384]
    
    # Structural embedding (NEW - from this project)
    focus_node = self.find_most_relevant_memory(input_text)
    struct_emb = self.structural_encoder.get_embedding(focus_node)  # [64]
    
    # Global context (NEW)
    all_struct = self.structural_encoder.encode_all(...)
    global_ctx = all_struct.mean(dim=0)  # [64]
    
    # Combine
    state = torch.cat([text_emb, struct_emb, global_ctx, density_features])
    return state  # [516]
```

---

## Timeline Summary

| Phase | Days | Key Deliverable |
|-------|------|-----------------|
| Foundation | 1-2 | View extraction working |
| Data Processing | 3-5 | Training pairs generated |
| Training System | 6-9 | Model converges |
| Evaluation | 10-11 | Embeddings validated |
| Production | 12-14 | Memory R1 integrated |

**Total: 14 days to production-ready GraphSAGE for Memory R1.**

---

## Appendix: Default Configuration

```yaml
# config/default.yaml

# Feature extraction
features:
  dimensions: 7
  include_entity_features: true

# Random walks
walks:
  length: 80
  per_node: 10
  context_window: 10

# Negative sampling
negatives:
  per_positive: 5
  exponent: 0.75

# Model
model:
  hidden_dim: 64
  output_dim: 64
  num_layers: 2
  dropout: 0.3
  aggregator: mean

# Training
training:
  learning_rate: 0.001
  batch_size: 512
  epochs: 100
  gradient_clip: 1.0
  early_stopping_patience: 10

# Inference
inference:
  use_compile: true
  cache_embeddings: true
```

---

## References

1. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE)
2. Mikolov, T., et al. (2013). Distributed Representations of Words and Phrases (Skip-gram)
3. Grover, A., & Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks
4. Neo4j Graph Data Science Library Documentation
5. Memory R1: Towards Grounded and Persistent Personalization through Reasoning with Working Memory
