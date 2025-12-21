"""Adaptive Router Training Module.

This module provides a TOML-based training script for Adaptive Router profiles.
It supports hybrid model loading (API fetch or TOML definition) and multiple
output formats (local, S3, MinIO).

Usage:
    python -m train.train --config config.toml

Or directly:
    python train/train.py --config config.toml
"""

from train.config import TrainingConfig, load_config
from train.train import train_router, main

__all__ = ["TrainingConfig", "load_config", "train_router", "main"]
