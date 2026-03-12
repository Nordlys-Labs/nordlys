"""Shared deterministic embedder and dataset conversion helpers for tests."""

import numpy as np
import pandas as pd

from nordlys import Dataset
from nordlys.embeddings import Embedder


class FakeEmbedder(Embedder):
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % (2**32)
            rng = np.random.default_rng(seed)
            group = seed % 8
            center = np.zeros(384, dtype=np.float32)
            center[group] = 10.0
            noise = rng.normal(0.0, 0.05, size=384).astype(np.float32)
            vectors.append(center + noise)
        return np.asarray(vectors, dtype=np.float32)

    def checkpoint_config(self) -> dict[str, str | bool]:
        return {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "trust_remote_code": False,
        }


def _to_dataset(df: pd.DataFrame, models: list[str]) -> Dataset:
    rows = []
    for idx, row in df.iterrows():
        best_model = max(models, key=lambda mid: float(row[mid]))
        targets = {mid: int(mid == best_model) for mid in models}
        rows.append(
            {"id": str(idx), "input": str(row["questions"]), "targets": targets}
        )
    return Dataset.from_list(rows)
