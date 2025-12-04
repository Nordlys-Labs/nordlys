#!/usr/bin/env python3
"""One-time migration script for RouterProfile v1 to v2 format.

This script migrates existing profiles with the following changes:
1. Merges llm_profiles dict into models list as error_rates field
2. Restructures metadata: moves lambda_min, lambda_max, and
   default_cost_preference into a nested routing config

After migration, the v1 format is no longer supported.

Usage:
    # Local files only
    python scripts/migrate_profile.py input.json output.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_router.models.storage import RouterProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_profile_v1_to_v2(data: dict) -> dict:
    """Migrate v1 profile to v2 format (one-time operation).

    Args:
        data: Raw profile dict with v1 schema

    Returns:
        Migrated profile dict with v2 schema
    """
    if "llm_profiles" not in data:
        logger.warning("Profile appears to already be in v2 format")
        return data

    logger.info("Migrating profile from v1 to v2 format...")

    models = data.get("models", [])
    llm_profiles = data.get("llm_profiles", {})

    # Merge error_rates into each model
    for model in models:
        provider = model["provider"].lower()
        model_name = model["model_name"].lower()
        model_id = f"{provider}/{model_name}"

        # Transfer error_rates from llm_profiles
        model["error_rates"] = llm_profiles.get(model_id, [])

    # Remove llm_profiles
    migrated = data.copy()
    migrated.pop("llm_profiles", None)

    # Migrate metadata routing config
    metadata = migrated.get("metadata", {})
    routing_config = {
        "lambda_min": metadata.pop("lambda_min", 0.0),
        "lambda_max": metadata.pop("lambda_max", 2.0),
        "default_cost_preference": metadata.pop("default_cost_preference", 0.5),
    }
    metadata["routing"] = routing_config
    migrated["metadata"] = metadata

    logger.info(f"Migration complete: {len(models)} models updated")
    return migrated


def load_data(path: str) -> dict:
    """Load profile data from local JSON file."""
    logger.info(f"Loading from file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_data(data: dict, path: str) -> None:
    """Save profile data to local JSON file."""
    # Validate before saving
    profile = RouterProfile(**data)

    logger.info(f"Saving to file: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile.model_dump(), f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="One-time migration: RouterProfile v1 to v2 (local files only)"
    )
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("output", help="Output JSON file path")
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate, don't save"
    )

    args = parser.parse_args()

    try:
        # Load old format
        logger.info("Loading profile...")
        old_data = load_data(args.input)

        # Migrate
        logger.info("Migrating schema...")
        new_data = migrate_profile_v1_to_v2(old_data)

        # Validate
        logger.info("Validating new schema...")
        profile = RouterProfile(**new_data)
        logger.info(f"✓ Validation successful: {len(profile.models)} models")

        if args.validate_only:
            logger.info("Validation-only mode, skipping save")
            return

        # Save
        logger.info("Saving migrated profile...")
        save_data(new_data, args.output)
        logger.info("✓ Migration complete")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
