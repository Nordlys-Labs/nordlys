"""Tests for embedders and Trainer embedder integration."""

from __future__ import annotations

import numpy as np

from nordlys import Dataset, Trainer
from nordlys.clustering import KMeansClusterer
from nordlys.embeddings import SentenceTransformers


class TestSentenceTransformers:
    def test_encode_returns_float32(self, monkeypatch):
        class FakeEncoder:
            def __init__(self, model, device, trust_remote_code):
                self.model = model
                self.device = device
                self.trust_remote_code = trust_remote_code
                self.tokenizer = type("T", (), {"clean_up_tokenization_spaces": True})()

            def encode(self, texts, **kwargs):
                assert kwargs["batch_size"] == 16
                return np.ones((len(texts), 3), dtype=np.float64)

        monkeypatch.setattr(
            "nordlys.embeddings.sentence_transformers.SentenceTransformer",
            FakeEncoder,
        )

        embedder = SentenceTransformers(
            model="fake-model",
            batch_size=16,
            normalize=False,
            device="cpu",
        )

        vectors = embedder.encode(["a", "b"])
        assert vectors.dtype == np.float32
        assert vectors.shape == (2, 3)
        assert embedder.checkpoint_config() == {
            "model": "fake-model",
            "trust_remote_code": False,
        }


class TestTrainerEmbedder:
    def test_default_make_embedder_uses_trainer_config(self, monkeypatch):
        captured: dict[str, object] = {}

        class FakeEmbedder:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def encode(self, texts: list[str]) -> np.ndarray:
                return np.zeros((len(texts), 4), dtype=np.float32)

            def checkpoint_config(self) -> dict[str, str | bool]:
                return {"model": "fake", "trust_remote_code": False}

        monkeypatch.setattr("nordlys.trainer.SentenceTransformers", FakeEmbedder)

        trainer = Trainer(
            models=["gpt-4"],
            embedding_model="my-model",
            embedding_batch_size=32,
            embedding_normalize=False,
            allow_trust_remote_code=True,
            device="cpu",
        )

        trainer._make_embedder()
        assert captured == {
            "model": "my-model",
            "batch_size": 32,
            "normalize": False,
            "trust_remote_code": True,
            "device": "cpu",
        }

    def test_custom_embedder_is_used_in_fit(self):
        class FakeEmbedder:
            def __init__(self):
                self.calls = 0

            def encode(self, texts: list[str]) -> np.ndarray:
                self.calls += 1
                return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

            def checkpoint_config(self) -> dict[str, str | bool]:
                return {"model": "fake-embedder", "trust_remote_code": False}

        dataset = Dataset.from_list(
            [
                {"id": "1", "input": "hello", "targets": {"gpt-4": 1}},
                {"id": "2", "input": "world", "targets": {"gpt-4": 0}},
            ]
        )
        embedder = FakeEmbedder()
        trainer = Trainer(
            models=["gpt-4"],
            embedder=embedder,
            clusterer=KMeansClusterer(n_clusters=1, random_state=42),
        )

        checkpoint = trainer.fit(dataset)

        assert checkpoint is not None
        assert embedder.calls == 1
