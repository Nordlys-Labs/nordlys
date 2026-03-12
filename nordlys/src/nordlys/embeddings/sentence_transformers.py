"""SentenceTransformers embedder."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from nordlys.embeddings.base import Embedder


class SentenceTransformers(Embedder):
    """Embedder powered by sentence-transformers.

    Exposes common sentence-transformers options plus pass-through kwargs.
    """

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 64,
        normalize: bool = True,
        trust_remote_code: bool = False,
        device: Literal["cpu", "cuda"] = "cpu",
        show_progress_bar: bool = False,
        prompt_name: str | None = None,
        prompt: str | None = None,
        max_seq_length: int | None = None,
        truncate_dim: int | None = None,
        revision: str | None = None,
        token: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize SentenceTransformers embedder.

        Args:
            model: Model name or path
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            trust_remote_code: Allow loading code from model repo
            device: Device to run on (cpu/cuda)
            show_progress_bar: Show progress during encoding
            prompt_name: Prompt name for the model (if supported)
            prompt: Custom prompt string to prepend
            max_seq_length: Maximum sequence length
            truncate_dim: Dimension to truncate to
            revision: Model revision
            token: HuggingFace token
            model_kwargs: Additional kwargs passed to model constructor
            tokenizer_kwargs: Additional kwargs passed to tokenizer
            config_kwargs: Additional kwargs passed to config
            encode_kwargs: Additional kwargs passed to encode()
        """
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.show_progress_bar = show_progress_bar
        self.prompt_name = prompt_name
        self.prompt = prompt
        self.max_seq_length = max_seq_length
        self.truncate_dim = truncate_dim
        self.revision = revision
        self.token = token
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.config_kwargs = config_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}

        # Build kwargs for SentenceTransformer constructor
        constructor_kwargs: dict[str, Any] = {
            "device": device,
            "trust_remote_code": trust_remote_code,
        }
        if revision:
            constructor_kwargs["revision"] = revision
        if token:
            constructor_kwargs["token"] = token
        if model_kwargs:
            constructor_kwargs.update(model_kwargs)

        self._encoder = SentenceTransformer(
            model,
            **constructor_kwargs,
        )

        # Configure tokenizer
        self._encoder.tokenizer.clean_up_tokenization_spaces = False

        # Store max_seq_length for encoding
        self._max_seq_length = max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of input texts

        Returns:
            2D numpy array of shape (len(texts), embedding_dim)
        """
        # Build encode kwargs
        encode_kwargs: dict[str, Any] = {
            "convert_to_numpy": True,
            "show_progress_bar": self.show_progress_bar,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize,
        }

        # Add prompt support
        if self.prompt_name:
            encode_kwargs["prompt_name"] = self.prompt_name
        if self.prompt:
            encode_kwargs["prompt"] = self.prompt

        # Add truncate_dim if provided
        if self.truncate_dim:
            encode_kwargs["truncate_dim"] = self.truncate_dim

        # Add max_seq_length if provided (via truncation)
        if self._max_seq_length:
            encode_kwargs["truncate"] = True

        # Merge user-provided encode kwargs
        encode_kwargs.update(self.encode_kwargs)

        vectors = self._encoder.encode(texts, **encode_kwargs)
        return np.asarray(vectors, dtype=np.float32)

    def checkpoint_config(self) -> dict[str, Any]:
        """Return embedding config stored in checkpoints."""
        config: dict[str, Any] = {
            "model": self.model,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.prompt_name:
            config["prompt_name"] = self.prompt_name
        if self.prompt:
            config["prompt"] = self.prompt
        if self.max_seq_length:
            config["max_seq_length"] = self.max_seq_length
        if self.truncate_dim:
            config["truncate_dim"] = self.truncate_dim
        if self.revision:
            config["revision"] = self.revision

        return config
