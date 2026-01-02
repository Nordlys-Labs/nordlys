"""Dimensionality reduction components for Nordlys."""

from nordlys.reduction._base import Reducer
from nordlys.reduction._pca import PCAReducer
from nordlys.reduction._umap import UMAPReducer

__all__ = ["Reducer", "UMAPReducer", "PCAReducer"]

