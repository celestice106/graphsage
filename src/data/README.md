# Data Module

This module handles all data loading, processing, and preparation for GraphSAGE training.

## Components

### graph_loader.py
- **GraphLoader**: Unified interface for loading graphs from various sources
- **MockGraphStore**: Mock graph store for testing and development
- Supports loading from Memory R1, JSON files, or creating synthetic data

### view_extractor.py
- **GraphSAGEViewExtractor**: Extracts memory-only view from full heterogeneous graph
- **GraphSAGEView**: Container for extracted view data
- Handles node re-indexing and edge filtering

### feature_extractor.py
- **MemoryFeatureExtractor**: Computes 7-dimensional features for memory nodes
- Features capture both structural and entity-derived information
- Applies log normalization to prevent feature dominance

### dataset.py
- **GraphSAGEDataset**: Complete dataset wrapper for training
- **create_data_object**: Creates PyTorch Geometric Data objects
- Supports saving/loading datasets to disk

## Feature Design

Each memory node has 7 features:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | caused_by_degree | Continuous | Number of causal connections |
| 1 | next_event_degree | Continuous | Number of temporal connections |
| 2 | num_entities_mentioned | Continuous | Entity count for this memory |
| 3 | shared_entity_neighbors | Continuous | Memories sharing entities |
| 4 | is_cause | Binary | Has outgoing caused_by edge |
| 5 | is_effect | Binary | Has incoming caused_by edge |
| 6 | has_successor | Binary | Has outgoing next_event edge |

## Usage

### Quick Start with Mock Data

```python
from src.data import GraphSAGEDataset

# Create dataset from mock graph
dataset = GraphSAGEDataset.from_mock(
    num_memories=500,
    num_entities=100,
    seed=42
)

# Get PyTorch Geometric data object
data = dataset.get_data(device=torch.device('cuda'))
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.shape[1]}")
print(f"Features: {data.x.shape}")
```

### From Saved File

```python
# Save dataset
dataset.save('data/processed/my_dataset')

# Load later
dataset = GraphSAGEDataset.load('data/processed/my_dataset')
```

### From Memory R1

```python
from memory_r1_bank import MemoryR1Bank
from src.data import GraphSAGEDataset

# Load from Memory R1 bank
bank = MemoryR1Bank()
# ... populate bank with memories ...

dataset = GraphSAGEDataset.from_memory_r1(bank)
```

## Data Flow

```
Full Graph (Memory R1)
        │
        ▼
  ┌─────────────────┐
  │  GraphLoader    │  Load from various sources
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  ViewExtractor  │  Extract memory-only view
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │FeatureExtractor │  Compute 7-dim features
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ GraphSAGEDataset│  Package for training
  └─────────────────┘
```
