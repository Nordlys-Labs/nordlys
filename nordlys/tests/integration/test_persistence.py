"""Integration tests for Router checkpoint loading."""

from json import JSONDecodeError

import pytest

from nordlys import Router


class TestCheckpointLoad:
    def test_construct_from_json_path(self, fitted_nordlys, tmp_path):
        path = tmp_path / "router.json"
        fitted_nordlys._checkpoint.to_json_file(str(path))

        loaded = Router(checkpoint=path)

        assert loaded is not None
        assert loaded._core_engine is not None

    def test_construct_from_msgpack_path(self, fitted_nordlys, tmp_path):
        path = tmp_path / "router.msgpack"
        fitted_nordlys._checkpoint.to_msgpack_file(str(path))

        loaded = Router(checkpoint=path)

        assert loaded is not None
        assert loaded._core_engine is not None

    def test_loaded_router_routes_identically(self, fitted_nordlys, tmp_path):
        path = tmp_path / "router.json"
        fitted_nordlys._checkpoint.to_json_file(str(path))

        before = fitted_nordlys.route("What is machine learning?")
        after = Router(checkpoint=path).route("What is machine learning?")

        assert before.model_id == after.model_id
        assert before.cluster_id == after.cluster_id


class TestCheckpointLoadErrors:
    def test_load_nonexistent_file_raises(self):
        with pytest.raises(RuntimeError, match="Failed to open checkpoint file"):
            Router(checkpoint="nonexistent_file.json")

    def test_load_corrupted_json_raises(self, tmp_path):
        path = tmp_path / "corrupted.json"
        path.write_text("{ invalid json")

        with pytest.raises(ValueError):
            Router(checkpoint=path)

    def test_load_corrupted_json_is_not_jsondecodeerror(self, tmp_path):
        path = tmp_path / "corrupted.json"
        path.write_text("{ invalid json")

        with pytest.raises(ValueError):
            try:
                Router(checkpoint=path)
            except JSONDecodeError:
                pytest.fail(
                    "Expected RuntimeError from core loader, not JSONDecodeError"
                )
