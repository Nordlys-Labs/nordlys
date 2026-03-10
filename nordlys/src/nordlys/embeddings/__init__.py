"""Embedders for Trainer."""

from nordlys.embeddings.base import Embedder
from nordlys.embeddings.sentence_transformers_backend import (
    SentenceTransformers,
)

__all__ = ["Embedder", "SentenceTransformers"]
