"""
Configuration module for GraphSAGE training.

This module provides configuration loading and validation utilities.
"""

from pathlib import Path
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default.yaml

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    return load_config()


__all__ = ['load_config', 'get_default_config']
