"""Writers registry and factory functions."""

from pathlib import Path
from typing import Type

from .base import ProfileWriter
from .csv import CSVProfileWriter
from .json import JSONProfileWriter
from .parquet import ParquetProfileWriter

# Registry mapping extensions to writer classes
_WRITER_REGISTRY: dict[str, Type[ProfileWriter]] = {}

# Auto-register writers
for writer_cls in [JSONProfileWriter, CSVProfileWriter, ParquetProfileWriter]:
    for ext in writer_cls.supported_extensions():
        ext = ext.lower().strip()
        _WRITER_REGISTRY[ext] = writer_cls  # type: ignore[type-abstract]


def get_writer(path: Path | str) -> ProfileWriter:
    """Get appropriate writer for file extension.

    Args:
        path: File path or extension

    Returns:
        Instantiated ProfileWriter

    Raises:
        ValueError: If extension not supported
    """
    ext = Path(path).suffix.lower()
    writer_cls = _WRITER_REGISTRY.get(ext)

    if not writer_cls:
        supported = ", ".join(_WRITER_REGISTRY.keys())
        raise ValueError(f"Unsupported format: {ext}. Supported formats: {supported}")

    return writer_cls()


def register_writer(writer_cls: Type[ProfileWriter]) -> None:
    """Register a custom writer class.

    Allows users to add support for custom formats.

    Args:
        writer_cls: Writer class to register
    """
    for ext in writer_cls.supported_extensions():
        ext = ext.lower().strip()
        _WRITER_REGISTRY[ext] = writer_cls


def supported_formats() -> list[str]:
    """Get list of supported file extensions.

    Returns:
        List of supported extensions
    """
    return list(_WRITER_REGISTRY.keys())
