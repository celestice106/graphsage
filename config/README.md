# Configuration Module

This folder contains YAML configuration files for GraphSAGE training.

## Files

- **default.yaml**: Default configuration with balanced settings. Use for development and experimentation.
- **production.yaml**: Production-optimized settings. Use for final training and deployment.

## Usage

```python
from config import load_config, get_default_config

# Load default config
config = get_default_config()

# Load specific config file
config = load_config('config/production.yaml')

# Access settings
batch_size = config['training']['batch_size']
hidden_dim = config['model']['hidden_dim']
```

## Key Settings

### Features
- `dimensions`: Number of input features per node (7 for Memory R1)
- `include_entity_features`: Whether to use entity-derived features

### Random Walks
- `length`: Walk length (80 recommended)
- `per_node`: Walks per node (10 recommended)
- `context_window`: Skip-gram context window (10 recommended)

### Model
- `hidden_dim`: Hidden layer size (64)
- `output_dim`: Embedding dimension (64)
- `num_layers`: Number of GraphSAGE layers (2)
- `dropout`: Dropout rate (0.3 for training, 0.0 for inference)

### Training
- `learning_rate`: Adam learning rate (0.001)
- `batch_size`: Positive pairs per batch (512-1024)
- `epochs`: Maximum epochs (100)
- `use_amp`: Enable mixed precision (false for stability)
- `use_compile`: Enable torch.compile (true for production)

## Customization

Create a new YAML file based on default.yaml to customize settings for your use case.
