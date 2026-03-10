"""SentenceTransformers embedder."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from nordlys.embeddings.base import Embedder


class SentenceTransformers(Embedder):
    """Embedder powered by sentence-transformers."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        batch_size: int = 64,
        normalize: bool = True,
        trust_remote_code: bool = False,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.trust_remote_code = trust_remote_code
        self.device = device
        self._encoder = SentenceTransformer(
            model,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        self._encoder.tokenizer.clean_up_tokenization_spaces = False

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "trust_remote_code": self.trust_remote_code,
        }
