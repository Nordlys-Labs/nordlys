"""Dimensionality reduction components for Router."""

from nordlys.reduction.base import Reducer, register_reducer, restore_reducer
from nordlys.reduction.pca import PCAReducer
from nordlys.reduction.umap_reducer import UMAPReducer

__all__ = [
    "Reducer",
    "PCAReducer",
    "UMAPReducer",
    "register_reducer",
    "restore_reducer",
]
