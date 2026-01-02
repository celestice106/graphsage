# Walks Module

This module implements the random walk co-occurrence approach for generating training data. It's the core of the GDS-style GraphSAGE training.

## Components

### generator.py
- **RandomWalkGenerator**: Basic random walk generation
- **BiasedRandomWalkGenerator**: node2vec-style biased walks with p/q parameters

### pair_sampler.py
- **CooccurrencePairSampler**: Extract (target, context) pairs from walks
- **SubsampledPairSampler**: With frequency-based subsampling

### negative_sampler.py
- **DegreeBiasedNegativeSampler**: Degree-biased sampling (recommended)
- **UniformNegativeSampler**: Uniform random sampling

## Key Concept

The training data generation follows this pipeline:

```
Graph Structure
      │
      ▼
Random Walks ────► [n1, n2, n3, n4, n5, ...]
      │
      ▼
Pair Extraction ─► [(n1,n2), (n1,n3), (n2,n1), (n2,n3), ...]
      │
      ▼
Negative Sampling ─► For each (target, context), sample negatives
      │
      ▼
Training Batch ──► (targets, contexts, negatives)
```

## Usage

### Basic Walk Generation

```python
from src.walks import RandomWalkGenerator

walker = RandomWalkGenerator(
    edge_index=edge_index,  # [2, num_edges]
    num_nodes=num_nodes,
    walk_length=80,
    walks_per_node=10,
    seed=42
)

walks = walker.generate_all_walks()
print(f"Generated {len(walks)} walks")
```

### Pair Extraction

```python
from src.walks import CooccurrencePairSampler

sampler = CooccurrencePairSampler(context_window=10)
pairs = sampler.extract_pairs(walks)

# Or get as tensors directly
targets, contexts = sampler.extract_pairs_tensor(walks, device='cuda')
```

### Negative Sampling

```python
from src.walks import DegreeBiasedNegativeSampler

neg_sampler = DegreeBiasedNegativeSampler(
    edge_index=edge_index,
    num_nodes=num_nodes,
    exponent=0.75,  # Sublinear dampening
    device='cuda'
)

# Sample 5 negatives for each positive pair
negatives = neg_sampler.sample(num_samples=len(pairs), num_negatives=5)
```

## Training Signal Amplification

With default parameters on a 500-node graph:

```
walks_per_node = 10
walk_length = 80
context_window = 10
nodes = 500

Positive pairs ≈ 500 × 10 × 80 × 10 = 4,000,000 pairs
```

Compare to link prediction: ~1,000-2,000 edges = ~2,000 pairs.

This provides **~2000x more training signal** from the same graph!

## Parameters

### Walk Parameters
- `walk_length`: Longer walks capture more global structure (default: 80)
- `walks_per_node`: More walks = more training data (default: 10)

### Context Parameters
- `context_window`: Larger windows = more pairs but weaker signal (default: 10)

### Negative Sampling Parameters
- `num_negatives`: More negatives = better discrimination (default: 5)
- `exponent`: 0.75 (Word2Vec default) dampens high-degree node influence
