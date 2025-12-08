"""Readers registry and factory functions."""

from pathlib import Path
from typing import Type

from .base import ProfileReader
from .csv import CSVProfileReader
from .json import JSONProfileReader
from .parquet import ParquetProfileReader

# Registry mapping extensions to reader classes
_READER_REGISTRY: dict[str, Type[ProfileReader]] = {}

# Auto-register readers
for reader_cls in [JSONProfileReader, CSVProfileReader, ParquetProfileReader]:
    for ext in reader_cls.supported_extensions():
        _READER_REGISTRY[ext] = reader_cls  # type: ignore[type-abstract]


def get_reader(path: Path | str) -> ProfileReader:
    """Get appropriate reader for file extension.

    Args:
        path: File path or extension

    Returns:
        Instantiated ProfileReader

    Raises:
        ValueError: If extension not supported
    """
    ext = Path(path).suffix.lower()
    reader_cls = _READER_REGISTRY.get(ext)

    if not reader_cls:
        supported = ", ".join(_READER_REGISTRY.keys())
        raise ValueError(f"Unsupported format: {ext}. Supported formats: {supported}")

    return reader_cls()


def register_reader(reader_cls: Type[ProfileReader]) -> None:
    """Register a custom reader class.

    Allows users to add support for custom formats.

    Args:
        reader_cls: Reader class to register
    """
    for ext in reader_cls.supported_extensions():
        _READER_REGISTRY[ext] = reader_cls


def supported_formats() -> list[str]:
    """Get list of supported file extensions.

    Returns:
        List of supported extensions
    """
    return list(_READER_REGISTRY.keys())
