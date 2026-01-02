# GraphSAGE Training for Memory R1 Structural Embeddings

GDS-Style Random Walk Co-occurrence Approach for learning structural embeddings in the Memory R1 system.

## Overview

This project implements a complete GraphSAGE training pipeline for learning structural embeddings of memory nodes. The embeddings capture graph topology and are used alongside text embeddings in Memory R1's dual embedding architecture.

### Key Features

- **Random Walk Co-occurrence**: Uses GDS-style skip-gram objective instead of link prediction
- **Training Signal Amplification**: Generates millions of pairs from small graphs
- **Production Ready**: Optimized for GPU training and sub-millisecond inference
- **Memory R1 Compatible**: Designed for seamless integration

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd graphsage_training_w_gds_style

# Install dependencies
pip install -r requirements.txt
```

### For Google Colab

```python
# Install PyTorch Geometric (if not already installed)
!pip install torch-geometric

# Clone and install
!git clone <repository_url>
%cd graphsage_training_w_gds_style
!pip install -r requirements.txt
```

### Training

```bash
# Generate training data
python scripts/generate_data.py --num-memories 500 --num-entities 100

# Train model
python scripts/train.py --config config/default.yaml

# Evaluate embeddings
python scripts/evaluate.py --checkpoint checkpoints/model_best.pt --visualize

# Export for production
python scripts/export_model.py --checkpoint checkpoints/model_best.pt
```

## Project Structure

```
graphsage_training_w_gds_style/
├── config/                    # Configuration files
│   ├── default.yaml          # Default hyperparameters
│   └── production.yaml       # Production settings
├── src/
│   ├── data/                 # Data processing
│   │   ├── graph_loader.py   # Load graphs from various sources
│   │   ├── view_extractor.py # Extract GraphSAGE view
│   │   ├── feature_extractor.py # Compute node features
│   │   └── dataset.py        # PyTorch dataset wrapper
│   ├── walks/                # Random walk generation
│   │   ├── generator.py      # Walk generation
│   │   ├── pair_sampler.py   # Co-occurrence pairs
│   │   └── negative_sampler.py # Negative sampling
│   ├── model/                # Model architecture
│   │   ├── graphsage.py      # ProductionGraphSAGE
│   │   └── loss.py           # Skip-gram loss
│   ├── training/             # Training pipeline
│   │   ├── trainer.py        # Main trainer
│   │   ├── batch_generator.py # Batch generation
│   │   └── callbacks.py      # Early stopping, checkpoints
│   ├── inference/            # Production inference
│   │   ├── encoder.py        # MemoryR1StructuralEncoder
│   │   └── cache.py          # Embedding cache
│   └── utils/                # Utilities
│       ├── metrics.py        # Evaluation metrics
│       └── visualization.py  # Plotting
├── scripts/                  # Command-line scripts
├── tests/                    # Test suite
└── notebooks/                # Jupyter notebooks
```

## Training Pipeline

```
Full Graph (Memory R1)
        │
        ▼
┌───────────────────┐
│ View Extraction   │  Memory nodes only, memory-to-memory edges
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Feature Extraction│  7-dimensional features per node
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Random Walks      │  10 walks × 80 steps per node
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Pair Extraction   │  Skip-gram (target, context) pairs
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ GraphSAGE Training│  2-layer model, skip-gram loss
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Production Export │  64-dim embeddings
└───────────────────┘
```

## Model Architecture

```
Input Features [N, 7]
       │
       ▼
SAGEConv(7, 64) + ReLU + Dropout(0.3)
       │
       ▼
SAGEConv(64, 64)
       │
       ▼
L2 Normalization
       │
       ▼
Output Embeddings [N, 64]
```

## Feature Design

Each memory node has 7 features:

| Feature | Type | Description |
|---------|------|-------------|
| caused_by_degree | Continuous | Causal connections |
| next_event_degree | Continuous | Temporal connections |
| num_entities_mentioned | Continuous | Entity richness |
| shared_entity_neighbors | Continuous | Co-reference potential |
| is_cause | Binary | Has outgoing caused_by |
| is_effect | Binary | Has incoming caused_by |
| has_successor | Binary | Has outgoing next_event |

## Memory R1 Integration

```python
from src.inference import MemoryR1StructuralEncoder

# Initialize encoder
encoder = MemoryR1StructuralEncoder(
    model_path='exports/graphsage_production.pt',
    device='cuda'
)

# Get embeddings for all memory nodes
embeddings = encoder.encode_all(full_graph)

# Get embedding for specific memory
emb = encoder.encode_by_id('mem_0001', full_graph)

# Invalidate cache when graph changes
encoder.invalidate_cache()
```

## Configuration

Key parameters in `config/default.yaml`:

```yaml
features:
  dimensions: 7

walks:
  length: 80
  per_node: 10
  context_window: 10

model:
  hidden_dim: 64
  output_dim: 64
  num_layers: 2
  dropout: 0.3

training:
  learning_rate: 0.001
  batch_size: 512
  epochs: 100
  early_stopping_patience: 10
```

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Training Loss | < 1.0 | Final skip-gram loss |
| Neighbor Similarity Gap | > 0.1 | Connected vs random pair similarity |
| Link Prediction AUC | > 0.75 | Edge prediction from embeddings |
| Inference Latency | < 5ms | Full graph embedding time |

## GPU Training

The project is optimized for GPU training:

```python
# All tensors should be on the same CUDA device
device = torch.device('cuda')
model = model.to(device)
features = features.to(device)
edge_index = edge_index.to(device)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=src
```

## Troubleshooting

### Training Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Loss stuck at ~1.4 | No learning | Check gradients, increase LR |
| All embeddings identical | Feature collapse | Verify feature variance |
| Loss oscillating | LR too high | Reduce LR, add gradient clipping |

### Memory Issues

- Reduce batch_size if OOM
- Use StreamingBatchGenerator for large pair sets
- Reduce walks_per_node for very large graphs

## References

1. Hamilton et al. (2017) - Inductive Representation Learning on Large Graphs (GraphSAGE)
2. Mikolov et al. (2013) - Distributed Representations of Words and Phrases (Skip-gram)
3. Neo4j Graph Data Science Library Documentation
4. Memory R1 Architecture Document

## License

[Specify your license]
