"""Convert router profile between JSON and MessagePack formats."""

import argparse
import json
import sys
from pathlib import Path

from adaptive_core_ext import RouterProfile


def get_output_path(input_path: Path, output: Path | None) -> Path:
    """Determine output path based on input format."""
    if output:
        return output
    if input_path.suffix.lower() == ".json":
        return input_path.with_suffix(".msgpack")
    return input_path.with_suffix(".json")


def load_profile(path: Path) -> RouterProfile:
    """Load profile from JSON or MessagePack based on extension."""
    if path.suffix.lower() == ".msgpack":
        return RouterProfile.from_msgpack_file(str(path))
    return RouterProfile.from_json_file(str(path))


def save_profile(profile: RouterProfile, path: Path, compact: bool = False) -> None:
    """Save profile to JSON or MessagePack based on extension."""
    if path.suffix.lower() == ".msgpack":
        profile.to_msgpack_file(str(path))
    else:
        # For pretty-printed JSON, we need to re-parse with Python json
        if compact:
            profile.to_json_file(str(path))
        else:
            json_str = profile.to_json_string()
            data = json.loads(json_str)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)


def get_model_count(profile: RouterProfile) -> int:
    """Get the number of models in the profile."""
    # The C++ core doesn't expose models count directly, so we parse JSON
    json_str = profile.to_json_string()
    data = json.loads(json_str)
    return len(data.get("models", []))


def profile_to_data(profile: RouterProfile) -> dict:
    """Return a canonical Python dict representation of a profile."""
    json_str = profile.to_json_string()
    return json.loads(json_str)


def validate_conversion(original: RouterProfile, output_path: Path) -> None:
    """Ensure the converted profile matches the original profile."""
    converted = load_profile(output_path)
    converted.validate()
    original_data = profile_to_data(original)
    converted_data = profile_to_data(converted)
    if original_data != converted_data:
        raise ValueError(
            "Conversion validation failed: output profile differs from input profile"
        )


def print_summary(profile: RouterProfile, input_path: Path, output_path: Path) -> None:
    """Print conversion summary."""
    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    ratio = output_size / input_size if input_size > 0 else 0
    model_count = get_model_count(profile)

    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Clusters: {profile.n_clusters}")
    print(f"  Models: {model_count}")
    print(f"  Dtype: {profile.dtype}")
    print(f"  Embedding model: {profile.embedding_model}")
    print(f"  Input size: {input_size:,} bytes")
    print(f"  Output size: {output_size:,} bytes ({ratio:.1%})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert router profile between JSON and MessagePack formats"
    )
    parser.add_argument("input", type=Path, help="Input profile (.json or .msgpack)")
    parser.add_argument("output", type=Path, nargs="?", help="Output file (optional)")
    parser.add_argument("--compact", action="store_true", help="Compact JSON output")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress summary")
    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Determine output path
    output_path = get_output_path(args.input, args.output)

    # Check same format
    if args.input.suffix.lower() == output_path.suffix.lower():
        print("Error: Input and output have same format", file=sys.stderr)
        return 1

    # Check overwrite
    if output_path.exists() and not args.force:
        print(
            f"Error: Output exists: {output_path} (use -f to overwrite)",
            file=sys.stderr,
        )
        return 1

    # Load, validate, save
    try:
        profile = load_profile(args.input)
        profile.validate()
        save_profile(profile, output_path, args.compact)
        validate_conversion(profile, output_path)

        if not args.quiet:
            print_summary(profile, args.input, output_path)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
