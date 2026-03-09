"""Tests for Dataset class."""

import pandas as pd
import polars as pl
import pytest

from nordlys.dataset import Dataset


class TestDatasetCreation:
    """Test dataset creation from different sources."""

    def test_from_list_empty(self):
        """Test creating empty dataset."""
        dataset = Dataset.from_list([])
        assert dataset.num_rows == 0
        assert dataset.column_names == []

    def test_from_list_single_row(self):
        """Test creating dataset from single row."""
        rows = [{"id": "1", "input": "test prompt"}]
        dataset = Dataset.from_list(rows)

        assert dataset.num_rows == 1
        # Column order may vary, just check presence
        assert set(dataset.column_names) == {"id", "input"}
        assert dataset.column("id") == ["1"]
        assert dataset.column("input") == ["test prompt"]

    def test_from_list_multiple_rows(self):
        """Test creating dataset from multiple rows."""
        rows = [
            {"id": "1", "input": "prompt 1"},
            {"id": "2", "input": "prompt 2"},
            {"id": "3", "input": "prompt 3"},
        ]
        dataset = Dataset.from_list(rows)

        assert dataset.num_rows == 3
        assert dataset.column("id") == ["1", "2", "3"]
        assert dataset.column("input") == ["prompt 1", "prompt 2", "prompt 3"]

    def test_from_list_with_extra_columns(self):
        """Test creating dataset with extra columns."""
        rows = [
            {"id": "1", "input": "prompt", "targets": {"gpt-4": 1}, "score": 0.9},
            {"id": "2", "input": "prompt", "targets": {"gpt-4": 0}, "score": 0.3},
        ]
        dataset = Dataset.from_list(rows)

        assert dataset.num_rows == 2
        # Column order may vary, just check presence
        assert set(dataset.column_names) == {"id", "input", "targets", "score"}
        assert dataset.column("targets") == [{"gpt-4": 1}, {"gpt-4": 0}]
        assert dataset.column("score") == [0.9, 0.3]

    def test_from_list_sparse_rows(self):
        """Test that from_list handles sparse rows by filling with None."""
        rows = [
            {"id": "1", "input": "prompt 1"},
            {"id": "2"},  # missing input - filled with None
        ]
        dataset = Dataset.from_list(rows)

        assert dataset.num_rows == 2
        assert dataset.column("id") == ["1", "2"]
        assert dataset.column("input") == ["prompt 1", None]

    def test_from_pandas(self):
        """Test creating dataset from pandas DataFrame."""
        df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "input": ["prompt 1", "prompt 2", "prompt 3"],
            }
        )

        dataset = Dataset.from_pandas(df)

        assert dataset.num_rows == 3
        assert dataset.column("id") == ["1", "2", "3"]
        assert dataset.column("input") == ["prompt 1", "prompt 2", "prompt 3"]

    def test_to_pandas_roundtrip(self):
        """Test that to_pandas preserves data."""
        rows = [
            {"id": "1", "input": "prompt 1", "extra": "a"},
            {"id": "2", "input": "prompt 2", "extra": "b"},
        ]
        original = Dataset.from_list(rows)
        df = original.to_pandas()
        restored = Dataset.from_pandas(df)

        assert restored.num_rows == original.num_rows
        assert restored.column_names == original.column_names
        assert restored.column("id") == original.column("id")
        assert restored.column("input") == original.column("input")


class TestDatasetProperties:
    """Test dataset properties and accessors."""

    def test_column(self):
        """Test getting a column."""
        rows = [{"id": "1", "input": "test"}]
        dataset = Dataset.from_list(rows)

        assert dataset.column("id") == ["1"]

    def test_column_not_found(self):
        """Test that missing column raises KeyError."""
        rows = [{"id": "1", "input": "test"}]
        dataset = Dataset.from_list(rows)

        with pytest.raises(KeyError, match="Column 'missing' not found"):
            dataset.column("missing")

    def test_has_column(self):
        """Test checking column existence."""
        rows = [{"id": "1", "input": "test"}]
        dataset = Dataset.from_list(rows)

        assert dataset.has_column("id") is True
        assert dataset.has_column("missing") is False

    def test_len(self):
        """Test len() works."""
        rows = [{"id": str(i)} for i in range(10)]
        dataset = Dataset.from_list(rows)

        assert len(dataset) == 10

    def test_getitem(self):
        """Test single row access."""
        rows = [
            {"id": "1", "input": "first"},
            {"id": "2", "input": "second"},
        ]
        dataset = Dataset.from_list(rows)

        assert dataset[0] == {"id": "1", "input": "first"}
        assert dataset[1] == {"id": "2", "input": "second"}

    def test_getitem_out_of_range(self):
        """Test that out-of-range index raises IndexError."""
        rows = [{"id": "1"}]
        dataset = Dataset.from_list(rows)

        with pytest.raises(IndexError):
            _ = dataset[10]

    def test_repr(self):
        """Test string representation."""
        rows = [{"id": "1", "input": "test"}]
        dataset = Dataset.from_list(rows)

        assert "Dataset" in repr(dataset)
        assert "num_rows=1" in repr(dataset)
        assert "id" in repr(dataset)


class TestDatasetMap:
    """Test dataset map transformations."""

    def test_map_single_row(self):
        """Test map with single-row transforms."""
        rows = [{"id": "1", "input": "test"}]

        def add_target(row):
            return {**row, "targets": {"gpt-4": 1}}

        dataset = Dataset.from_list(rows)
        result = dataset.map(add_target)

        assert result.num_rows == 1
        assert result.column("targets") == [{"gpt-4": 1}]

    def test_map_multiple_columns(self):
        """Test map can add multiple columns."""
        rows = [{"id": "1", "input": "test"}]

        def transform(row):
            return {
                "targets": {"gpt-4": 1},
                "extra": "value",
            }

        dataset = Dataset.from_list(rows)
        result = dataset.map(transform)

        # Check that original columns are preserved and new ones added
        assert set(result.column_names) == {"id", "input", "targets", "extra"}
        assert result.column("extra") == ["value"]

    def test_map_batched(self):
        """Test batched map."""
        rows = [{"id": str(i), "input": f"prompt {i}"} for i in range(10)]

        def batch_transform(batch):
            # Just add a constant value - easier to verify
            return [{**row, "targets": {"gpt-4": 1}} for row in batch]

        dataset = Dataset.from_list(rows)
        result = dataset.map(batch_transform, batched=True, batch_size=5)

        assert result.num_rows == 10
        assert result.column("targets") == [{"gpt-4": 1}] * 10

    def test_map_batched_requires_batch_size(self):
        """Test that batched map requires batch_size."""
        rows = [{"id": "1"}]

        def transform(batch):
            return batch

        dataset = Dataset.from_list(rows)

        with pytest.raises(ValueError, match="batch_size is required"):
            dataset.map(transform, batched=True)

    def test_map_preserves_original(self):
        """Test that map returns new dataset, doesn't modify original."""
        rows = [{"id": "1", "input": "test"}]
        dataset = Dataset.from_list(rows)

        def add_column(row):
            return {**row, "new": "value"}

        result = dataset.map(add_column)

        assert dataset.num_rows == 1
        assert not dataset.has_column("new")
        assert result.num_rows == 1
        assert result.has_column("new")


class TestDatasetFilter:
    """Test dataset filtering."""

    def test_filter(self):
        """Test basic filtering."""
        rows = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.3},
            {"id": "3", "score": 0.7},
        ]
        dataset = Dataset.from_list(rows)

        result = dataset.filter(lambda row: row["score"] > 0.5)

        assert result.num_rows == 2
        assert result.column("id") == ["1", "3"]

    def test_filter_all_kept(self):
        """Test filter where all rows match."""
        rows = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.8},
        ]
        dataset = Dataset.from_list(rows)

        result = dataset.filter(lambda row: row["score"] > 0.5)

        assert result.num_rows == 2

    def test_filter_none_kept(self):
        """Test filter where no rows match."""
        rows = [
            {"id": "1", "score": 0.3},
            {"id": "2", "score": 0.2},
        ]
        dataset = Dataset.from_list(rows)

        result = dataset.filter(lambda row: row["score"] > 0.5)

        assert result.num_rows == 0


class TestDatasetSelect:
    """Test dataset selection."""

    def test_select(self):
        """Test selecting specific indices."""
        rows = [{"id": str(i)} for i in range(5)]
        dataset = Dataset.from_list(rows)

        result = dataset.select([0, 2, 4])

        assert result.num_rows == 3
        assert result.column("id") == ["0", "2", "4"]

    def test_select_empty(self):
        """Test selecting empty list."""
        rows = [{"id": "1"}, {"id": "2"}]
        dataset = Dataset.from_list(rows)

        result = dataset.select([])

        assert result.num_rows == 0


class TestDatasetShuffle:
    """Test dataset shuffling."""

    def test_shuffle_changes_order(self):
        """Test that shuffle changes row order."""
        rows = [{"id": str(i)} for i in range(100)]
        dataset = Dataset.from_list(rows)

        shuffled = dataset.shuffle(seed=42)

        assert shuffled.num_rows == 100
        assert shuffled.column("id") != dataset.column("id")

    def test_shuffle_reproducible(self):
        """Test that shuffle with seed is reproducible."""
        rows = [{"id": str(i)} for i in range(10)]
        dataset = Dataset.from_list(rows)

        shuffled1 = dataset.shuffle(seed=42)
        shuffled2 = dataset.shuffle(seed=42)

        assert shuffled1.column("id") == shuffled2.column("id")


class TestDatasetTrainTestSplit:
    """Test train/test splitting."""

    def test_train_test_split(self):
        """Test basic train/test split."""
        rows = [{"id": str(i)} for i in range(10)]
        dataset = Dataset.from_list(rows)

        train, test = dataset.train_test_split(test_size=0.2, seed=42)

        assert train.num_rows == 8
        assert test.num_rows == 2

    def test_train_test_split_reproducible(self):
        """Test that split with seed is reproducible."""
        rows = [{"id": str(i)} for i in range(100)]
        dataset = Dataset.from_list(rows)

        train1, test1 = dataset.train_test_split(test_size=0.2, seed=42)
        train2, test2 = dataset.train_test_split(test_size=0.2, seed=42)

        assert train1.column("id") == train2.column("id")
        assert test1.column("id") == test2.column("id")


class TestDatasetValidateSchema:
    """Test schema validation."""

    def test_validate_default_schema(self):
        """Test validation with default required columns."""
        rows = [{"id": "1", "input": "test"}]
        dataset = Dataset.from_list(rows)

        missing = dataset.validate_schema()

        assert missing == []

    def test_validate_missing_columns(self):
        """Test validation detects missing columns."""
        rows = [{"id": "1"}]  # missing input
        dataset = Dataset.from_list(rows)

        missing = dataset.validate_schema()

        assert len(missing) > 0
        assert any("input" in err for err in missing)

    def test_validate_custom_columns(self):
        """Test validation with custom required columns."""
        rows = [{"id": "1", "input": "test", "custom": "value"}]
        dataset = Dataset.from_list(rows)

        missing = dataset.validate_schema(required_columns=["id", "custom"])

        assert missing == []


class TestDatasetPolars:
    """Test Polars DataFrame interop."""

    def test_from_polars(self):
        """Test creating dataset from Polars DataFrame."""
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "input": ["prompt 1", "prompt 2", "prompt 3"],
            }
        )

        dataset = Dataset.from_polars(df)

        assert dataset.num_rows == 3
        assert dataset.column("id") == ["1", "2", "3"]
        assert dataset.column("input") == ["prompt 1", "prompt 2", "prompt 3"]

    def test_to_polars(self):
        """Test converting dataset to Polars DataFrame."""
        rows = [
            {"id": "1", "input": "prompt 1"},
            {"id": "2", "input": "prompt 2"},
        ]
        dataset = Dataset.from_list(rows)

        df = dataset.to_polars()

        assert isinstance(df, pl.DataFrame)
        assert df.height == 2
        assert df.columns == ["id", "input"]

    def test_polars_roundtrip(self):
        """Test Polars roundtrip preserves data."""
        rows = [
            {"id": "1", "input": "prompt 1", "extra": "a"},
            {"id": "2", "input": "prompt 2", "extra": "b"},
        ]
        original = Dataset.from_list(rows)
        df = original.to_polars()
        restored = Dataset.from_polars(df)

        assert restored.num_rows == original.num_rows
        assert set(restored.column_names) == set(original.column_names)
        assert restored.column("id") == original.column("id")
        assert restored.column("input") == original.column("input")

    def test_pandas_polars_parity(self):
        """Test that pandas and polars produce equivalent results."""
        data = {
            "id": ["1", "2", "3"],
            "input": ["a", "b", "c"],
        }

        pdf = pd.DataFrame(data)
        pldf = pl.DataFrame(data)

        ds_pandas = Dataset.from_pandas(pdf)
        ds_polars = Dataset.from_polars(pldf)

        assert ds_pandas.num_rows == ds_polars.num_rows
        assert ds_pandas.column("id") == ds_polars.column("id")
        assert ds_pandas.column("input") == ds_polars.column("input")


class TestDatasetMapPreservesColumns:
    """Test that map preserves existing columns."""

    def test_map_partial_update(self):
        """Test that map preserves existing columns when adding new ones."""
        rows = [
            {"id": "1", "input": "test", "extra": "keep"},
            {"id": "2", "input": "test2", "extra": "keep2"},
        ]
        dataset = Dataset.from_list(rows)

        def add_targets(row):
            return {"targets": {"gpt-4": 1}}

        result = dataset.map(add_targets)

        assert result.num_rows == 2
        # Original columns preserved
        assert set(result.column_names) == {"id", "input", "extra", "targets"}
        assert result.column("id") == ["1", "2"]
        assert result.column("input") == ["test", "test2"]
        assert result.column("extra") == ["keep", "keep2"]
        # New column added
        assert result.column("targets") == [{"gpt-4": 1}, {"gpt-4": 1}]

    def test_map_batched_preserves_columns(self):
        """Test that batched map preserves existing columns."""
        rows = [{"id": str(i), "input": f"prompt {i}"} for i in range(5)]
        dataset = Dataset.from_list(rows)

        def batch_transform(batch):
            return [{"targets": {"gpt-4": 1}} for _ in batch]

        result = dataset.map(batch_transform, batched=True, batch_size=2)

        assert result.num_rows == 5
        assert set(result.column_names) == {"id", "input", "targets"}

    def test_map_batched_wrong_length_raises(self):
        """Test that batched map with wrong output length raises."""
        rows = [{"id": str(i)} for i in range(5)]
        dataset = Dataset.from_list(rows)

        def bad_transform(batch):
            # Return wrong number of rows
            return [{"x": 1} for _ in range(len(batch) - 1)]

        with pytest.raises(ValueError, match="same number of rows"):
            dataset.map(bad_transform, batched=True, batch_size=3)


class TestDatasetEmptyOperations:
    """Test that empty operations preserve schema."""

    def test_empty_filter_preserves_schema(self):
        """Test that filter with no matches preserves schema."""
        rows = [{"id": "1", "input": "test"}]
        dataset = Dataset.from_list(rows)

        result = dataset.filter(lambda row: False)

        assert result.num_rows == 0
        # Schema preserved
        assert set(result.column_names) == {"id", "input"}

    def test_empty_select_preserves_schema(self):
        """Test that select with empty list preserves schema."""
        rows = [{"id": "1", "input": "test"}]
        dataset = Dataset.from_list(rows)

        result = dataset.select([])

        assert result.num_rows == 0
        # Schema preserved
        assert set(result.column_names) == {"id", "input"}


class TestDatasetTargetsValidation:
    """Test targets column validation."""

    def test_validate_targets_valid(self):
        """Test that valid targets pass validation."""
        rows = [
            {"id": "1", "input": "test", "targets": {"gpt-4": 1, "claude": 0}},
            {"id": "2", "input": "test", "targets": {"gpt-4": 0}},
        ]
        dataset = Dataset.from_list(rows)

        errors = dataset.validate_targets()

        assert errors == []

    def test_validate_targets_not_dict(self):
        """Test that non-dict targets fail validation."""
        rows = [
            {"id": "1", "input": "test", "targets": "not a dict"},
        ]
        dataset = Dataset.from_list(rows)

        errors = dataset.validate_targets()

        assert len(errors) > 0
        assert "must be a dict" in errors[0]

    def test_validate_targets_invalid_value(self):
        """Test that non-binary target values fail validation."""
        rows = [
            {"id": "1", "input": "test", "targets": {"gpt-4": 0.5}},
        ]
        dataset = Dataset.from_list(rows)

        errors = dataset.validate_targets()

        assert len(errors) > 0
        assert "must be 0 or 1" in errors[0]

    def test_validate_targets_missing_is_ok(self):
        """Test that missing targets (None) is allowed."""
        rows = [
            {"id": "1", "input": "test", "targets": None},
            {"id": "2", "input": "test"},
        ]
        dataset = Dataset.from_list(rows)

        errors = dataset.validate_targets()

        assert errors == []


class TestDatasetUniqueIds:
    """Test ID uniqueness validation."""

    def test_validate_unique_ids(self):
        """Test that duplicate IDs are detected."""
        rows = [
            {"id": "1", "input": "test"},
            {"id": "1", "input": "test2"},  # duplicate
        ]
        dataset = Dataset.from_list(rows)

        errors = dataset.validate_schema(check_unique_ids=True)

        assert len(errors) > 0
        assert "not unique" in errors[0]

    def test_validate_unique_ids_pass(self):
        """Test that unique IDs pass validation."""
        rows = [
            {"id": "1", "input": "test"},
            {"id": "2", "input": "test2"},
        ]
        dataset = Dataset.from_list(rows)

        errors = dataset.validate_schema(check_unique_ids=True)

        assert errors == []


class TestDatasetRemoveColumns:
    """Test remove_columns method."""

    def test_remove_columns(self):
        """Test removing specific columns."""
        rows = [
            {"id": "1", "input": "test", "extra1": "a", "extra2": "b"},
        ]
        dataset = Dataset.from_list(rows)

        result = dataset.remove_columns(["extra1"])

        assert set(result.column_names) == {"id", "input", "extra2"}
        assert result.column("id") == ["1"]

    def test_remove_columns_multiple(self):
        """Test removing multiple columns."""
        rows = [
            {"id": "1", "input": "test", "a": "1", "b": "2", "c": "3"},
        ]
        dataset = Dataset.from_list(rows)

        result = dataset.remove_columns(["a", "c"])

        assert set(result.column_names) == {"id", "input", "b"}
