# Training Module

This module implements the complete training pipeline for GraphSAGE with the skip-gram objective.

## Components

### trainer.py
- **GraphSAGETrainer**: Main trainer class
- **train_graphsage()**: High-level training function

### batch_generator.py
- **BatchGenerator**: Generate training batches from pairs
- **StreamingBatchGenerator**: Memory-efficient version for large datasets
- **TrainingData**: Container for all training data

### callbacks.py
- **EarlyStopping**: Stop when validation loss plateaus
- **ModelCheckpoint**: Save best model checkpoints
- **TrainingLogger**: Log metrics and progress

## Training Pipeline

```
Positive Pairs + Negative Sampler
              │
              ▼
       ┌─────────────┐
       │ BatchGenerator │ Shuffle, chunk, sample negatives
       └──────┬──────┘
              │
    ┌─────────▼─────────┐
    │  For each batch:   │
    │  1. Forward pass   │  embeddings = model(features, edge_index)
    │  2. Index batch    │  target_emb = embeddings[targets]
    │  3. Compute loss   │  loss = skip_gram_loss(target, context, neg)
    │  4. Backward       │  loss.backward()
    │  5. Clip & step    │  optimizer.step()
    └─────────┬─────────┘
              │
              ▼
       ┌─────────────┐
       │ Validation   │ Compute val loss, check early stopping
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │ Checkpoint   │ Save if best, periodic saves
       └─────────────┘
```

## Usage

### Basic Training

```python
from src.training import GraphSAGETrainer
from src.model import ProductionGraphSAGE
from src.walks import DegreeBiasedNegativeSampler, CooccurrencePairSampler

# Prepare data
pairs = pair_sampler.extract_pairs(walks)
neg_sampler = DegreeBiasedNegativeSampler(edge_index, num_nodes, device='cuda')

# Create model
model = ProductionGraphSAGE().cuda()

# Create trainer
trainer = GraphSAGETrainer(
    model=model,
    features=features.cuda(),
    edge_index=edge_index.cuda(),
    positive_pairs=pairs,
    negative_sampler=neg_sampler,
    config=config
)

# Train
best_loss = trainer.train(num_epochs=100)

# Get embeddings
embeddings = trainer.get_embeddings()
```

### High-Level API

```python
from src.training import train_graphsage

model, embeddings = train_graphsage(
    dataset=dataset,
    walks=walks,
    config=config,
    device=torch.device('cuda')
)
```

## Configuration

Key training parameters in config:

```yaml
training:
  learning_rate: 0.001      # Adam learning rate
  batch_size: 512           # Pairs per batch
  epochs: 100               # Max epochs
  gradient_clip: 1.0        # Gradient clipping norm
  early_stopping_patience: 10  # Epochs without improvement

validation:
  val_fraction: 0.1         # Fraction for validation

negatives:
  per_positive: 5           # Negatives per positive pair
```

## GPU Training

The trainer is designed for GPU-only training:
- All tensors must be on the same CUDA device
- Model should be moved to GPU before creating trainer
- Batch generation happens on GPU for efficiency

```python
device = torch.device('cuda')

model = ProductionGraphSAGE().to(device)
features = features.to(device)
edge_index = edge_index.to(device)
```

## Monitoring

Training progress is logged to:
- Console (every `log_every` epochs)
- `logs/epoch_metrics.json` (all metrics)
- `logs/training_summary.json` (final summary)

Checkpoints saved to:
- `checkpoints/model_best.pt` (best model)
- `checkpoints/model_epochN.pt` (periodic)
