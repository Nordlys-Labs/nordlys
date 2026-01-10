"""Convert router checkpoint between JSON and MessagePack formats."""

import argparse
import json
import sys
from pathlib import Path

from nordlys_core_ext import NordlysCheckpoint


def get_output_path(input_path: Path, output: Path | None) -> Path:
    """Determine output path based on input format."""
    if output:
        return output
    if input_path.suffix.lower() == ".json":
        return input_path.with_suffix(".msgpack")
    return input_path.with_suffix(".json")


def load_checkpoint(path: Path) -> NordlysCheckpoint:
    """Load checkpoint from JSON or MessagePack based on extension."""
    if path.suffix.lower() == ".msgpack":
        return NordlysCheckpoint.from_msgpack_file(str(path))
    return NordlysCheckpoint.from_json_file(str(path))


def save_checkpoint(
    checkpoint: NordlysCheckpoint, path: Path, compact: bool = False
) -> None:
    """Save checkpoint to JSON or MessagePack based on extension."""
    if path.suffix.lower() == ".msgpack":
        checkpoint.to_msgpack_file(str(path))
    else:
        # For pretty-printed JSON, we need to re-parse with Python json
        if compact:
            checkpoint.to_json_file(str(path))
        else:
            json_str = checkpoint.to_json_string()
            data = json.loads(json_str)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)


def get_model_count(checkpoint: NordlysCheckpoint) -> int:
    """Get the number of models in the checkpoint."""
    # The C++ core doesn't expose models count directly, so we parse JSON
    json_str = checkpoint.to_json_string()
    data = json.loads(json_str)
    return len(data.get("models", []))


def checkpoint_to_data(checkpoint: NordlysCheckpoint) -> dict:
    """Return a canonical Python dict representation of a checkpoint."""
    json_str = checkpoint.to_json_string()
    return json.loads(json_str)


def validate_conversion(original: NordlysCheckpoint, output_path: Path) -> None:
    """Ensure the converted checkpoint matches the original checkpoint."""
    converted = load_checkpoint(output_path)
    converted.validate()
    original_data = checkpoint_to_data(original)
    converted_data = checkpoint_to_data(converted)
    if original_data != converted_data:
        raise ValueError(
            "Conversion validation failed: output checkpoint differs from input checkpoint"
        )


def print_summary(
    checkpoint: NordlysCheckpoint, input_path: Path, output_path: Path
) -> None:
    """Print conversion summary."""
    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    ratio = output_size / input_size if input_size > 0 else 0
    model_count = get_model_count(checkpoint)

    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Clusters: {checkpoint.n_clusters}")
    print(f"  Models: {model_count}")
    print(f"  Dtype: {checkpoint.dtype}")
    print(f"  Embedding model: {checkpoint.embedding_model}")
    print(f"  Input size: {input_size:,} bytes")
    print(f"  Output size: {output_size:,} bytes ({ratio:.1%})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert router checkpoint between JSON and MessagePack formats"
    )
    parser.add_argument("input", type=Path, help="Input checkpoint (.json or .msgpack)")
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
        checkpoint = load_checkpoint(args.input)
        checkpoint.validate()
        save_checkpoint(checkpoint, output_path, args.compact)
        validate_conversion(checkpoint, output_path)

        if not args.quiet:
            print_summary(checkpoint, args.input, output_path)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
