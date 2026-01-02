# Model Module

This module implements the GraphSAGE model architecture and loss functions for learning structural embeddings.

## Components

### graphsage.py
- **ProductionGraphSAGE**: Main model for training and inference
- **FlexibleGraphSAGE**: Experimental version with more options
- **create_model()**: Factory function to create model from config

### loss.py
- **SkipGramLoss**: Main loss function (recommended)
- **MarginRankingLoss**: Alternative margin-based loss
- **InfoNCELoss**: Contrastive loss variant

### layers.py
- **SAGELayer**: Wrapped SAGEConv with batch norm and activation
- **L2NormLayer**: L2 normalization layer

## Architecture

```
Input Features [N, 7]
       │
       ▼
┌──────────────────┐
│ SAGEConv(7, 64)  │  Aggregate 1-hop neighbors
│ ReLU             │  Non-linearity
│ Dropout(0.3)     │  Regularization
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ SAGEConv(64, 64) │  Aggregate 2-hop neighbors
│ (no activation)  │  Linear output
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ L2 Normalize     │  Required for dot-product similarity
└────────┬─────────┘
         │
         ▼
Output Embeddings [N, 64]
```

## Skip-gram Loss

The loss function trains embeddings so that:
1. **Positive pairs** (from walks) have high dot-product similarity
2. **Negative pairs** (random) have low dot-product similarity

```
L = -log(σ(target · context)) - Σ log(σ(-target · negative))
```

Where σ is the sigmoid function.

## Usage

### Creating the Model

```python
from src.model import ProductionGraphSAGE

model = ProductionGraphSAGE(
    in_channels=7,      # 7 features per node
    hidden_channels=64, # Hidden dimension
    out_channels=64,    # Embedding dimension
    dropout=0.3,        # Regularization
)

# Move to GPU
model = model.cuda()
```

### Forward Pass

```python
import torch

# Input
features = torch.randn(100, 7).cuda()     # 100 nodes, 7 features
edge_index = torch.randint(0, 100, (2, 500)).cuda()  # 500 edges

# Get embeddings
embeddings = model(features, edge_index)
print(embeddings.shape)  # [100, 64]
print(embeddings.norm(dim=1))  # All ~1.0 (L2 normalized)
```

### Computing Loss

```python
from src.model import SkipGramLoss

loss_fn = SkipGramLoss()

# Batch of training data
targets = torch.tensor([0, 1, 2, 3]).cuda()
contexts = torch.tensor([5, 6, 7, 8]).cuda()
negatives = torch.tensor([[10, 11], [12, 13], [14, 15], [16, 17]]).cuda()

# Compute loss
loss = loss_fn(embeddings, targets, contexts, negatives)
loss.backward()
```

## Design Decisions

### Why L2 Normalization?
- Enables dot-product as similarity metric
- Prevents embedding norm from growing unboundedly
- Required for skip-gram objective to work properly

### Why Mean Aggregation?
- Most stable across different graph structures
- Works well for sparse graphs
- Empirically best for structural embeddings

### Why 2 Layers?
- Captures 2-hop neighborhood
- Sufficient for most graph patterns
- More layers can cause over-smoothing

### Why Dropout Only After First Layer?
- Regularizes hidden representations
- No dropout on output to get consistent embeddings
- Follows common practice in GNNs
