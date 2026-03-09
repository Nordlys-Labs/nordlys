"""Dataset abstraction for Nordlys.

A lightweight, column-oriented dataset similar to HuggingFace datasets,
but focused on the minimal functionality needed for routing workflows.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union, cast

import pandas as pd

T = TypeVar("T")


@dataclass
class Dataset:
    """Column-oriented dataset for routing.

    This is a lightweight HF-like dataset focused on the minimal functionality
    needed for training routers. It stores data in column-oriented format
    for performance.

    Required columns:
        - id: unique identifier for each row
        - input: the text input to route

    All other columns are optional user data.

    Example:
        >>> dataset = Dataset.from_list([
        ...     {"id": "1", "input": "Fix the parser", "targets": {"gpt-4": 1}},
        ...     {"id": "2", "input": "Write a poem", "targets": {"gpt-4": 0}},
        ... ])
        >>> dataset.num_rows
        2
    """

    _columns: dict[str, list[Any]] = field(default_factory=dict)
    _num_rows: int = field(default_factory=int)

    def __post_init__(self) -> None:
        if self._num_rows > 0:
            for col in self._columns.values():
                if len(col) != self._num_rows:
                    msg = f"Column length mismatch: expected {self._num_rows}, got {len(col)}"
                    raise ValueError(msg)

    @classmethod
    def from_list(cls, rows: list[dict[str, Any]]) -> Dataset:
        """Create dataset from a list of dictionaries.

        Args:
            rows: List of dictionaries, each representing a row.

        Returns:
            Dataset instance.

        Example:
            >>> dataset = Dataset.from_list([
            ...     {"id": "1", "input": "Fix the parser", "targets": {"gpt-4": 1}},
            ... ])
        """
        if not rows:
            return cls(_columns={}, _num_rows=0)

        columns: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            for key, value in row.items():
                columns[key].append(value)

        num_rows = len(rows)
        return cls(_columns=dict(columns), _num_rows=num_rows)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Dataset:
        """Create dataset from a pandas DataFrame.

        Args:
            df: pandas DataFrame.

        Returns:
            Dataset instance.

        Example:
            >>> df = pd.DataFrame({
            ...     "id": ["1", "2"],
            ...     "input": ["Fix parser", "Write poem"],
            ... })
            >>> dataset = Dataset.from_pandas(df)
        """
        columns = {col: df[col].tolist() for col in df.columns}
        return cls(_columns=columns, _num_rows=len(df))

    def to_pandas(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame.

        Returns:
            pandas DataFrame.

        Example:
            >>> df = dataset.to_pandas()
        """
        return pd.DataFrame(self._columns)

    @property
    def num_rows(self) -> int:
        """Number of rows in the dataset."""
        return self._num_rows

    @property
    def column_names(self) -> list[str]:
        """List of column names."""
        return list(self._columns.keys())

    def column(self, name: str) -> list[Any]:
        """Get a column by name.

        Args:
            name: Column name.

        Returns:
            List of values in the column.

        Raises:
            KeyError: If column doesn't exist.
        """
        if name not in self._columns:
            msg = f"Column '{name}' not found. Available: {self.column_names}"
            raise KeyError(msg)
        return self._columns[name]

    def has_column(self, name: str) -> bool:
        """Check if a column exists.

        Args:
            name: Column name.

        Returns:
            True if column exists.
        """
        return name in self._columns

    def map(
        self,
        function: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        batched: bool = False,
        batch_size: int | None = None,
    ) -> Dataset:
        """Apply a transformation function to the dataset.

        HF-style: the function receives a dict (single row) or a list of dicts
        (batched) and returns a dict with new or modified columns.

        Args:
            function: Transform function.
                - If batched=False: receives dict, returns dict
                - If batched=True: receives list[dict], returns list[dict]
            batched: Whether to apply in batch mode (recommended for perf).
            batch_size: Number of rows per batch. Required if batched=True.

        Returns:
            New Dataset with transformed data.

        Example:
            >>> def add_targets(example):
            ...     return {**example, "targets": {"gpt-4": 1}}
            >>> dataset = dataset.map(add_targets)

            >>> def add_targets_batched(batch):
            ...     return [{"targets": {"gpt-4": 1}} for _ in batch]
            >>> dataset = dataset.map(add_targets_batched, batched=True, batch_size=100)
        """
        if batched:
            if batch_size is None:
                msg = "batch_size is required when batched=True"
                raise ValueError(msg)
            # Cast to batch function type - runtime behavior is determined by batched flag
            batch_fn = cast(
                Callable[[list[dict[str, Any]]], list[dict[str, Any]]], function
            )
            return self._map_batched(batch_fn, batch_size)

        return self._map_single(function)

    def _map_single(
        self, function: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> Dataset:
        """Apply function row-by-row."""
        new_columns: dict[str, list[Any]] = defaultdict(list)

        for i in range(self._num_rows):
            row = {col: self._columns[col][i] for col in self._columns}
            transformed = function(row)
            for key, value in transformed.items():
                new_columns[key].append(value)

        return Dataset(_columns=dict(new_columns), _num_rows=self._num_rows)

    def _map_batched(
        self,
        function: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        batch_size: int,
    ) -> Dataset:
        """Apply function in batches."""
        new_columns: dict[str, list[Any]] = defaultdict(list)

        for start in range(0, self._num_rows, batch_size):
            end = min(start + batch_size, self._num_rows)
            batch = [
                {col: self._columns[col][i] for col in self._columns}
                for i in range(start, end)
            ]
            transformed_batch = function(batch)

            for transformed in transformed_batch:
                for key, value in transformed.items():
                    new_columns[key].append(value)

        return Dataset(_columns=dict(new_columns), _num_rows=self._num_rows)

    def filter(self, function: Callable[[dict[str, Any]], bool]) -> Dataset:
        """Filter rows based on a predicate function.

        Args:
            function: Predicate function that receives a row dict and returns bool.

        Returns:
            New Dataset with only rows where function returns True.

        Example:
            >>> dataset = dataset.filter(lambda row: row.get("score", 0) > 0.5)
        """
        kept_indices = []
        for i in range(self._num_rows):
            row = {col: self._columns[col][i] for col in self._columns}
            if function(row):
                kept_indices.append(i)

        if not kept_indices:
            return Dataset(_columns={}, _num_rows=0)

        new_columns = {
            col: [self._columns[col][i] for i in kept_indices] for col in self._columns
        }
        return Dataset(_columns=new_columns, _num_rows=len(kept_indices))

    def select(self, indices: list[int]) -> Dataset:
        """Select rows by index.

        Args:
            indices: List of row indices to keep.

        Returns:
            New Dataset with selected rows.

        Example:
            >>> dataset = dataset.select([0, 2, 4])
        """
        if not indices:
            return Dataset(_columns={}, _num_rows=0)

        new_columns = {
            col: [self._columns[col][i] for i in indices] for col in self._columns
        }
        return Dataset(_columns=new_columns, _num_rows=len(indices))

    def shuffle(self, seed: int | None = None) -> Dataset:
        """Shuffle the dataset.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            New Dataset with shuffled rows.
        """
        if seed is not None:
            random.seed(seed)

        indices = list(range(self._num_rows))
        random.shuffle(indices)

        new_columns = {
            col: [self._columns[col][i] for i in indices] for col in self._columns
        }
        return Dataset(_columns=new_columns, _num_rows=self._num_rows)

    def train_test_split(
        self,
        test_size: float = 0.2,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset]:
        """Split into train and test sets.

        Args:
            test_size: Fraction of data for test set (0.0 to 1.0).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, test_dataset).

        Example:
            >>> train, test = dataset.train_test_split(test_size=0.2, seed=42)
        """
        if seed is not None:
            random.seed(seed)

        indices = list(range(self._num_rows))
        random.shuffle(indices)

        split_idx = int(self._num_rows * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train = self.select(train_indices)
        test = self.select(test_indices)

        return train, test

    def validate_schema(
        self,
        required_columns: list[str] | None = None,
    ) -> list[str]:
        """Validate that required columns exist.

        Args:
            required_columns: List of required column names.
                Defaults to ["id", "input"].

        Returns:
            List of missing column names (empty if all present).

        Example:
            >>> missing = dataset.validate_schema()
            >>> if missing:
            ...     raise ValueError(f"Missing: {missing}")
        """
        if required_columns is None:
            required_columns = ["id", "input"]

        return [col for col in required_columns if col not in self._columns]

    def __repr__(self) -> str:
        return f"Dataset(num_rows={self._num_rows}, columns={self.column_names})"

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single row by index.

        Args:
            index: Row index.

        Returns:
            Dictionary representing the row.
        """
        if index < 0 or index >= self._num_rows:
            msg = f"Index {index} out of range for dataset with {self._num_rows} rows"
            raise IndexError(msg)

        return {col: self._columns[col][index] for col in self._columns}
