# Inference Module

This module provides inference components for computing structural embeddings in the Memory Bank system.

## Components

### encoder.py
- **StructuralEncoder**: Main encoder for Memory Bank integration
- **benchmark_encoder()**: Benchmark encoder performance

### cache.py
- **EmbeddingCache**: Full cache with invalidation support
- **LRUEmbeddingCache**: LRU cache for partial access patterns

## Usage

### Basic Inference

```python
from src.inference import StructuralEncoder

# Initialize encoder
encoder = StructuralEncoder(
    model_path='exports/graphsage.pt',
    device='cuda',
    cache_embeddings=True
)

# Get all embeddings
embeddings = encoder.encode_all(full_graph)
print(embeddings.shape)  # [num_nodes, 64]

# Get single embedding by original ID
emb = encoder.encode_by_id('mem_0001', full_graph)

# Get embedding by index
emb = encoder.encode_single(42, full_graph)
```

### Cache Management

```python
# Embeddings are cached automatically
emb1 = encoder.encode_all(full_graph)  # Computes
emb2 = encoder.encode_all(full_graph)  # Returns cached

# Invalidate when graph changes
encoder.invalidate_cache()  # Full invalidation
encoder.invalidate_cache(['mem_0001', 'mem_0002'])  # Partial

# Check cache statistics
stats = encoder.get_statistics()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
```

### Memory Bank Integration

```python
# In Memory Bank's structural embedding module:

from src.inference import StructuralEncoder

class StructuralEmbeddingModule:
    def __init__(self, model_path='exports/graphsage.pt'):
        self.encoder = StructuralEncoder(model_path)

    def get_embedding(self, memory_id, graph_store):
        return self.encoder.encode_by_id(memory_id, graph_store)

    def get_all_embeddings(self, graph_store):
        return self.encoder.encode_all(graph_store)

    def on_graph_change(self, changed_memory_ids):
        self.encoder.invalidate_cache(changed_memory_ids)
```

## Caching Strategy

The cache is designed for Memory Bank's use case:

1. **Full Invalidation**: Any graph change invalidates entire cache
   - GraphSAGE embeddings depend on neighborhood structure
   - Changing one node affects its neighbors' embeddings

2. **Lazy Recomputation**: Embeddings computed on first access after invalidation
   - No wasted computation if embeddings aren't needed immediately

3. **Optional Time-Based Expiry**: Can set maximum age for cache entries

## Performance

Target performance characteristics:
- **Latency**: < 5ms for full graph (< 1ms target with caching)
- **Throughput**: 200+ inferences/second
- **Memory**: Minimal overhead (~4 bytes per node × embedding_dim)

### Benchmarking

```python
from src.inference import benchmark_encoder

results = benchmark_encoder(
    encoder=encoder,
    full_graph=full_graph,
    num_iterations=100
)

print(f"Mean: {results['mean_ms']:.2f}ms")
print(f"Throughput: {results['throughput_per_sec']:.1f}/s")
```

## torch.compile Optimization

For additional speedup, enable torch.compile:

```python
encoder = StructuralEncoder(
    model_path='model.pt',
    use_compile=True  # Enable torch.compile
)
```

Note: First inference may be slower due to compilation overhead.
