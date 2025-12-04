"""Model resolution utilities for the Adaptive Router application.

These helpers operate directly on the models embedded inside the router
profile so that the FastAPI service no longer depends on the external
model registry.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, List

from adaptive_router.models.api import Model

logger = logging.getLogger(__name__)


def _normalise_identifier(value: str) -> str:
    """Normalize model identifier components for comparison."""
    return value.strip().lower()


def _build_model_indexes(
    available_models: Iterable[Model],
) -> tuple[dict[str, Model], dict[str, list[str]]]:
    """Create lookup dictionaries for available models."""
    models_by_id: dict[str, Model] = {}
    models_by_author: dict[str, list[str]] = defaultdict(list)

    for model in available_models:
        model_id = model.unique_id()
        if model_id in models_by_id:
            logger.warning(
                "Duplicate model '%s' detected in profile, using first entry", model_id
            )
            continue

        models_by_id[model_id] = model
        author_key = _normalise_identifier(model.provider)
        models_by_author[author_key].append(model_id)

    return models_by_id, models_by_author


def resolve_models(
    model_specs: List[str],
    available_models: List[Model],
) -> List[Model]:
    """Resolve user-specified model IDs against the loaded router profile.

    Args:
        model_specs: Model identifiers in "provider/model_name" format.
        available_models: Models embedded inside the router profile.

    Returns:
        List of Model objects referenced by ``model_specs`` in the order provided.

    Raises:
        ValueError: If a spec is malformed or the referenced model is unavailable.
    """
    if not available_models:
        raise ValueError("No models loaded in router profile")

    models_by_id, models_by_author = _build_model_indexes(available_models)
    skipped_specs: list[str] = []

    resolved_models: list[Model] = []
    for spec in model_specs:
        try:
            author, model_name = spec.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid model specification '{spec}': expected format 'author/model_name'"
            ) from exc

        author_key = _normalise_identifier(author)
        model_key = _normalise_identifier(model_name)
        lookup_key = f"{author_key}/{model_key}"

        model = models_by_id.get(lookup_key)
        if model is None:
            suggestions = models_by_author.get(author_key, [])
            suggestion_text = ""
            if suggestions:
                preview = ", ".join(suggestions[:5])
                suffix = (
                    ""
                    if len(suggestions) <= 5
                    else f" (and {len(suggestions) - 5} more)"
                )
                suggestion_text = f". Available {author} models: {preview}{suffix}"
            skipped_specs.append(f"{spec}{suggestion_text}")
            continue

        resolved_models.append(model)

    for skipped in skipped_specs:
        logger.warning("Requested model ignored: %s", skipped)

    if not resolved_models:
        skipped_msg = f" ({'; '.join(skipped_specs)})" if skipped_specs else ""
        raise ValueError(
            f"No requested models are available in the router profile{skipped_msg}"
        )

    return resolved_models
