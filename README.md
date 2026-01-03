# GraphSAGE Structural Embedding Training

A production-ready pipeline for training GraphSAGE models to generate structural embeddings for memory graphs. Designed for the Memory R1-style Memory Bank with dual-embedding system.

## Overview

This project trains a Graph Neural Network (GraphSAGE) to learn **structural embeddings** - vector representations that capture how nodes are connected in a graph. These embeddings complement text embeddings in a dual-embedding system that is used by Reinforcement Learning agents for enhancing memory retrieval.

### The Problem We Solve

Current AI chatbots are **stateless**—they forget everything once a user ends their session. While naive RAG (Retrieval-Augmented Generation) approaches can partially address this by storing and retrieving past conversations, they lack awareness of **temporal context**: information from the distant past may still be highly relevant to today's questions, but simple similarity-based retrieval cannot capture this.

Our solution introduces a **dual-embedding system** (structural + semantic) to represent each memory entry in a memory bank. Combined with two **RL-trained agents** that operate and distill memories across sessions (inspired by the [Memory R1 paper](https://arxiv.org/abs/2508.19828)), this architecture successfully overcomes the long-term memory problem. Structural embeddings enable agents to understand not just *what* a memory contains, but *where* and *when* it exists in the long-horizon conversation graph spanning multiple sessions.

In the Memory Bank, memories form a graph through relationships:
- `caused_by`: Memory A caused Memory B
- `next_event`: Memory A happened before Memory B
- `mention`: Memory mentions Entity E

Text embeddings capture **what** a memory says. Structural embeddings capture **where** a memory sits in the causal/temporal graph - enabling retrieval of structurally relevant memories even when text similarity is low.

GraphSAGE model learns to **encode** nodes' positional information into embeddings. Nodes that are close to each other in the graph may have similar embeddings, even if they are not directly connected.

### Our Approach: Random Walk Co-occurrence

Instead of traditional link prediction (which fails on sparse graphs), we use **GDS-style random walk co-occurrence**:

1. **Generate random walks** on the graph (like a drunk person wandering)
2. **Extract co-occurring pairs** from walks (nodes that appear near each other)
3. **Train with skip-gram objective** (like Word2Vec, but for graphs)

This amplifies training signal from ~2K edges to ~4M pairs on a 500-node graph.

## Quick Start

### Training on Google Colab

1. Upload this project to Colab or clone from GitHub
2. Open `notebooks/train_colab.ipynb`
3. Run all cells
4. Download exported model from `exports/`

### Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data and train
python scripts/train.py --config config/default.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Export for production
python scripts/export_model.py --checkpoint checkpoints/best_model.pt
```

## Project Structure

```
graphsage_training/
├── config/                 # Configuration files
│   ├── default.yaml       # Default hyperparameters
│   └── production.yaml    # Production settings
│
├── src/                   # Source code
│   ├── data/             # Graph loading & feature extraction
│   │   ├── graph_loader.py      # Load graphs from various sources
│   │   ├── view_extractor.py    # Extract memory-only subgraph
│   │   ├── feature_extractor.py # Compute 7-dim node features
│   │   └── dataset.py           # PyTorch Geometric dataset
│   │
│   ├── walks/            # Random walk generation
│   │   ├── generator.py         # Random walk generator
│   │   ├── pair_sampler.py      # Extract co-occurrence pairs
│   │   └── negative_sampler.py  # Degree-biased negative sampling
│   │
│   ├── model/            # GraphSAGE model
│   │   ├── graphsage.py         # Main model architecture
│   │   ├── layers.py            # Custom layers
│   │   └── loss.py              # Skip-gram & other losses
│   │
│   ├── training/         # Training pipeline
│   │   ├── trainer.py           # Main training loop
│   │   ├── batch_generator.py   # Batch creation
│   │   └── callbacks.py         # Early stopping, checkpoints
│   │
│   ├── inference/        # Production inference
│   │   ├── encoder.py           # Memory R1 encoder interface
│   │   └── cache.py             # Embedding cache
│   │
│   └── utils/            # Utilities
│       ├── metrics.py           # Evaluation metrics
│       ├── visualization.py     # t-SNE, plots
│       └── graph_utils.py       # Graph utilities
│
├── scripts/              # Runnable scripts
│   ├── train.py         # Training entry point
│   ├── evaluate.py      # Evaluation script
│   └── export_model.py  # Export for production
│
├── notebooks/            # Jupyter notebooks
│   ├── train_colab.ipynb       # Colab training notebook
│   └── lesson_graphsage.ipynb  # Educational notebook
│
└── tests/               # Unit tests
```

## Key Concepts

### 1. Node Features (7 dimensions)

Each memory node has 7 computed features:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `caused_by_degree` | Number of incoming causal edges |
| 1 | `next_event_degree` | Number of temporal sequence edges |
| 2 | `num_entities` | Entities mentioned by this memory |
| 3 | `shared_entity_neighbors` | Neighbors sharing entities |
| 4 | `is_cause` | Binary: has outgoing causal edges |
| 5 | `is_effect` | Binary: has incoming causal edges |
| 6 | `has_successor` | Binary: has temporal successors |

### 2. GraphSAGE Architecture

```
Input Features [N, 7]
       │
       ▼
SAGEConv(7 → 64) + ReLU + Dropout
       │
       ▼
SAGEConv(64 → 64)
       │
       ▼
L2 Normalization
       │
       ▼
Output Embeddings [N, 64]
```

- **2 layers** = captures 2-hop neighborhood structure
- **Mean aggregation** = stable and effective
- **L2 normalization** = enables dot-product similarity

### 3. Skip-gram Loss with Scale Factor

The critical insight that made training work:

```python
# Problem: L2-normalized embeddings have dot products in [-1, 1]
# sigmoid(1.0) = 0.73  ← Can't push higher!
# sigmoid(-1.0) = 0.27 ← Can't push lower!

# Solution: Scale before sigmoid
pos_loss = -log(sigmoid(dot_product * 5.0))  # Now sigmoid can reach 0.99
neg_loss = -log(sigmoid(-dot_product * 5.0)) # Now sigmoid can reach 0.01
```

Without scaling, training metrics plateau with `pos_prob ~0.71`, `neg_prob ~0.50`.

### 4. Training Signal Amplification

| Approach | Training Pairs | Problem |
|----------|---------------|---------|
| Link Prediction | ~2,000 (edges) | Too sparse |
| Random Walk Co-occurrence | ~4,000,000 | Rich signal! |

From 500 nodes × 10 walks × 80 length × ~10 context = millions of pairs.

## Configuration

Key hyperparameters in `config/default.yaml`:

```yaml
walks:
  length: 80          # Steps per walk
  per_node: 10        # Walks starting from each node
  context_window: 10  # Co-occurrence window size

model:
  hidden_dim: 64      # Hidden layer size
  output_dim: 64      # Embedding dimension
  num_layers: 2       # GraphSAGE layers
  dropout: 0.3        # Regularization

training:
  learning_rate: 0.001
  batch_size: 512
  epochs: 100
  early_stopping_patience: 10

negatives:
  per_positive: 5     # Negative samples per positive pair
  exponent: 0.75      # Degree bias (Word2Vec default)
```

## Evaluation Metrics

After training, we evaluate embedding quality:

| Metric | Good Value | Meaning |
|--------|------------|---------|
| Neighbor Similarity Gap | > 0.1 | Connected nodes more similar than random |
| Link Prediction AUC | > 0.7 | Can predict edges from embeddings |
| Avg Precision | > 0.7 | Precision-recall quality |
| Is Normalized | True | Embeddings are unit vectors |
| Is Collapsed | False | Embeddings aren't all identical |

Our results: **AUC = 0.978**, **Gap = 0.66** (excellent!)

## Inference Endpoint

The encoder supports multiple input formats for easy integration:

```python
from scripts.encode import GraphSAGEEncoder

# Load trained model
encoder = GraphSAGEEncoder('exports/graphsage_production.pt')

# === Option 1: PyTorch Geometric ===
from torch_geometric.data import Data
data = Data(x=features, edge_index=edges)
embeddings = encoder.encode(data)

# === Option 2: NetworkX ===
import networkx as nx
G = nx.karate_club_graph()
embeddings = encoder.encode(G)

# With node ID mapping preserved:
emb_dict = encoder.encode_networkx(G)  # {node_id: embedding}

# === Option 3: Dict format ===
graph = {'edge_index': edges, 'num_nodes': 100}
embeddings = encoder.encode(graph)

# === Option 4: Edge list + features ===
embeddings = encoder.encode((edge_index, features))
```

**Supported Input Formats:**
| Format | Features | Notes |
|--------|----------|-------|
| PyG Data | `data.x` | Native format, fastest |
| NetworkX | Node attributes or auto-computed | Easy integration |
| Dict | `'x'` or `'features'` key | Flexible |
| Tuple | Second element | `(edge_index, features)` |

If no features are provided, structural features (in/out degree, centrality, etc.) are computed automatically.

## Integration with Memory Bank 

```python
from scripts.encode import GraphSAGEEncoder

# Load trained model
encoder = GraphSAGEEncoder('exports/graphsage_production.pt', device='cuda')

# Compute structural embeddings for all memories
structural_emb = encoder.encode(memory_graph)

# Dual embedding retrieval
text_emb = text_encoder(query)
combined_score = alpha * (query @ text_emb.T) + (1-alpha) * (query_struct @ structural_emb.T)
```

## Requirements

```
torch>=2.0.0
torch-geometric>=2.3.0
pyyaml>=6.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

## License

MIT License - See LICENSE file.

## Acknowledgments

- GraphSAGE: [Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)
- Node2Vec random walks: [Grover & Leskovec, 2016](https://arxiv.org/abs/1607.00653)
- Skip-gram objective: [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)
- Memory R1: [Sikuan Yan et al., 2025](https://arxiv.org/abs/2508.19828)
