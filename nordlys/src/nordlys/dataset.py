"""Dataset abstraction for Router.

A lightweight, column-oriented dataset similar to HuggingFace datasets,
but focused on the minimal functionality needed for routing workflows.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

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

        Missing keys in individual rows are filled with None.

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

        # Collect all keys across all rows
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())

        # Build columns, filling missing keys with None
        columns: dict[str, list[Any]] = {key: [] for key in all_keys}
        for row in rows:
            for key in all_keys:
                columns[key].append(row.get(key))

        num_rows = len(rows)
        return cls(_columns=columns, _num_rows=num_rows)

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

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> Dataset:
        """Create dataset from a Polars DataFrame.

        Args:
            df: Polars DataFrame.

        Returns:
            Dataset instance.

        Example:
            >>> df = pl.DataFrame({
            ...     "id": ["1", "2"],
            ...     "input": ["Fix parser", "Write poem"],
            ... })
            >>> dataset = Dataset.from_polars(df)
        """
        columns = {col: df[col].to_list() for col in df.columns}
        return cls(_columns=columns, _num_rows=len(df))

    def to_pandas(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame.

        Returns:
            pandas DataFrame.

        Example:
            >>> df = dataset.to_pandas()
        """
        return pd.DataFrame(self._columns)

    def to_polars(self) -> pl.DataFrame:
        """Convert dataset to Polars DataFrame.

        Returns:
            Polars DataFrame.

        Example:
            >>> df = dataset.to_polars()
        """
        return pl.DataFrame(self._columns)

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

        Returns a copy to avoid mutation of internal state.

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
        # Return a copy to avoid exposing mutable internals
        return list(self._columns[name])

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
        (batched) and returns updates that are merged with existing columns.

        Unlike HF, this preserves existing columns - only add/update columns
        returned by the function. To drop columns, use remove_columns().

        Args:
            function: Transform function.
                - If batched=False: receives dict, returns dict with updates
                - If batched=True: receives list[dict], returns list[dict] with updates
            batched: Whether to apply in batch mode (recommended for perf).
            batch_size: Number of rows per batch. Required if batched=True.

        Returns:
            New Dataset with transformed data.

        Example:
            >>> def add_targets(example):
            ...     return {"targets": {"gpt-4": 1}}
            >>> dataset = dataset.map(add_targets)

            >>> def add_targets_batched(batch):
            ...     return [{"targets": {"gpt-4": 1}} for _ in batch]
            >>> dataset = dataset.map(add_targets_batched, batched=True, batch_size=100)
        """
        if batched:
            if batch_size is None:
                msg = "batch_size is required when batched=True"
                raise ValueError(msg)
            batch_fn = cast(
                Callable[[list[dict[str, Any]]], list[dict[str, Any]]], function
            )
            return self._map_batched(batch_fn, batch_size)

        return self._map_single(function)

    def _map_single(
        self, function: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> Dataset:
        """Apply function row-by-row, merging updates with existing columns."""
        # Start with existing columns
        new_columns: dict[str, list[Any]] = {
            col: list(self._columns[col]) for col in self._columns
        }

        # Ensure all expected columns exist in output
        for col in self._columns:
            if col not in new_columns:
                new_columns[col] = [None] * self._num_rows

        for i in range(self._num_rows):
            row = {col: self._columns[col][i] for col in self._columns}
            updates = function(row)

            # Merge updates - add new columns or update existing ones
            for key, value in updates.items():
                if key not in new_columns:
                    # New column - fill previous rows with None
                    new_columns[key] = [None] * i
                new_columns[key].append(value)

            # Fill any columns that weren't in this update
            for col in new_columns:
                if len(new_columns[col]) == i:
                    new_columns[col].append(None)

        # Trim to same length
        min_len = min(len(col) for col in new_columns.values()) if new_columns else 0
        for col in new_columns:
            new_columns[col] = new_columns[col][:min_len]

        return Dataset(_columns=new_columns, _num_rows=min_len)

    def _map_batched(
        self,
        function: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        batch_size: int,
    ) -> Dataset:
        """Apply function in batches, merging updates with existing columns."""
        # Start with existing columns
        new_columns: dict[str, list[Any]] = {
            col: list(self._columns[col]) for col in self._columns
        }

        for start in range(0, self._num_rows, batch_size):
            end = min(start + batch_size, self._num_rows)
            batch = [
                {col: self._columns[col][i] for col in self._columns}
                for i in range(start, end)
            ]
            batch_updates = function(batch)

            # Validate output length
            if len(batch_updates) != len(batch):
                msg = (
                    f"Batch transform must return same number of rows as input. "
                    f"Expected {len(batch)}, got {len(batch_updates)}"
                )
                raise ValueError(msg)

            for i, updates in enumerate(batch_updates):
                # Merge updates - add new columns or update existing ones
                for key, value in updates.items():
                    if key not in new_columns:
                        # New column - fill previous rows with None
                        new_columns[key] = [None] * (start + i)
                    new_columns[key].append(value)

                # Fill any columns that weren't in this update
                current_idx = start + i
                for col in new_columns:
                    if len(new_columns[col]) == current_idx:
                        new_columns[col].append(None)

        # Trim to same length
        min_len = min(len(col) for col in new_columns.values()) if new_columns else 0
        for col in new_columns:
            new_columns[col] = new_columns[col][:min_len]

        return Dataset(_columns=new_columns, _num_rows=min_len)

    def filter(self, function: Callable[[dict[str, Any]], bool]) -> Dataset:
        """Filter rows based on a predicate function.

        Preserves schema even if no rows match.

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
            # Return empty dataset with schema preserved
            return Dataset(_columns={col: [] for col in self._columns}, _num_rows=0)

        new_columns = {
            col: [self._columns[col][i] for i in kept_indices] for col in self._columns
        }
        return Dataset(_columns=new_columns, _num_rows=len(kept_indices))

    def select(self, indices: list[int]) -> Dataset:
        """Select rows by index.

        Preserves schema even if no indices selected.

        Args:
            indices: List of row indices to keep.

        Returns:
            New Dataset with selected rows.

        Example:
            >>> dataset = dataset.select([0, 2, 4])
        """
        if not indices:
            # Return empty dataset with schema preserved
            return Dataset(_columns={col: [] for col in self._columns}, _num_rows=0)

        # Validate all indices are in range
        min_idx = min(indices)
        max_idx = max(indices)
        if min_idx < 0:
            raise IndexError(
                f"Index {min_idx} out of range for Dataset with {self._num_rows} rows"
            )
        if max_idx >= self._num_rows:
            raise IndexError(
                f"Index {max_idx} out of range for Dataset with {self._num_rows} rows"
            )

        new_columns = {
            col: [self._columns[col][i] for i in indices] for col in self._columns
        }
        return Dataset(_columns=new_columns, _num_rows=len(indices))

    def shuffle(self, seed: int | None = None) -> Dataset:
        """Shuffle the dataset.

        Uses a local random generator to avoid mutating global state.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            New Dataset with shuffled rows.
        """
        rng = random.Random(seed)

        indices = list(range(self._num_rows))
        rng.shuffle(indices)

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

        Uses a local random generator to avoid mutating global state.

        Args:
            test_size: Fraction of data for test set (0.0 to 1.0).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, test_dataset).

        Example:
            >>> train, test = dataset.train_test_split(test_size=0.2, seed=42)
        """
        rng = random.Random(seed)

        indices = list(range(self._num_rows))
        rng.shuffle(indices)

        split_idx = int(self._num_rows * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train = self.select(train_indices)
        test = self.select(test_indices)

        return train, test

    def validate_schema(
        self,
        required_columns: list[str] | None = None,
        check_unique_ids: bool = False,
    ) -> list[str]:
        """Validate that required columns exist and optionally check IDs.

        Args:
            required_columns: List of required column names.
                Defaults to ["id", "input"].
            check_unique_ids: Whether to validate that IDs are unique.

        Returns:
            List of validation errors (empty if all pass).

        Example:
            >>> errors = dataset.validate_schema()
            >>> if errors:
            ...     raise ValueError(f"Validation failed: {errors}")
        """
        errors: list[str] = []

        if required_columns is None:
            required_columns = ["id", "input"]

        for col in required_columns:
            if col not in self._columns:
                errors.append(f"Missing required column: {col}")

        if check_unique_ids and "id" in self._columns:
            ids = self._columns["id"]
            if len(ids) != len(set(ids)):
                errors.append("IDs are not unique")

        return errors

    def validate_targets(self) -> list[str]:
        """Validate the optional targets column.

        Checks that targets, if present, is a dict of {model_id: 0|1}.

        Returns:
            List of validation errors (empty if all pass).
        """
        errors: list[str] = []

        if "targets" not in self._columns:
            return errors

        for i, targets in enumerate(self._columns["targets"]):
            if targets is None:
                continue

            if not isinstance(targets, dict):
                errors.append(
                    f"Row {i}: targets must be a dict, got {type(targets).__name__}"
                )
                continue

            for model_id, score in targets.items():
                if not isinstance(model_id, str):
                    errors.append(
                        f"Row {i}: model_id must be str, got {type(model_id).__name__}"
                    )
                if score not in (0, 1):
                    errors.append(f"Row {i}: target value must be 0 or 1, got {score}")

        return errors

    def remove_columns(self, columns: list[str]) -> Dataset:
        """Remove specific columns.

        Args:
            columns: List of column names to remove.

        Returns:
            New Dataset without the specified columns.

        Example:
            >>> dataset = dataset.remove_columns(["extra_field"])
        """
        new_columns = {
            col: list(self._columns[col]) for col in self._columns if col not in columns
        }
        return Dataset(_columns=new_columns, _num_rows=self._num_rows)

    def __repr__(self) -> str:
        return f"Dataset(num_rows={self._num_rows}, columns={self.column_names})"

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single row by index.

        Supports negative indices like -1 for the last row.

        Args:
            index: Row index (supports negative indices).

        Returns:
            Dictionary representing the row.
        """
        # Normalize negative indices
        if index < 0:
            index += self._num_rows

        if index < 0 or index >= self._num_rows:
            msg = f"Index {index} out of range for dataset with {self._num_rows} rows"
            raise IndexError(msg)

        return {col: self._columns[col][index] for col in self._columns}
