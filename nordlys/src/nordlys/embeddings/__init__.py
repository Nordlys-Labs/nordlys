"""Embedders for Trainer."""

from nordlys.embeddings.base import Embedder
from nordlys.embeddings.sentence_transformers import (
    SentenceTransformers,
)

__all__ = ["Embedder", "SentenceTransformers"]
