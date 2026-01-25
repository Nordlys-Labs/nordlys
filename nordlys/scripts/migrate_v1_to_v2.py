#!/usr/bin/env python3
"""Migrate checkpoint from v1 to v2 format.

v1 format:
- cluster_centers: {n_clusters, feature_dim, cluster_centers: [[...]]}
- models: [{provider, model_name, cost_per_1m_input_tokens, cost_per_1m_output_tokens, error_rates}]
- metadata: {allow_trust_remote_code, n_clusters, silhouette_score, embedding_model, embedding_dim, ...}

v2 format:
- version: "2.0"
- cluster_centers: [[...]]  (flat array)
- models: [{model_id, cost_per_1m_input_tokens, cost_per_1m_output_tokens, error_rates}]
- embedding: {model, dtype, trust_remote_code}
- clustering: {n_clusters, random_state, max_iter, n_init, algorithm, normalization}
- metrics: {n_samples, cluster_sizes, silhouette_score, inertia}
"""

import argparse
import json
import sys
from pathlib import Path


def detect_dtype(cluster_centers: list[list[float]]) -> str:
    """Detect dtype from cluster centers values."""
    if not cluster_centers or not cluster_centers[0]:
        return "float32"
    sample = cluster_centers[0][0]
    if isinstance(sample, float) and abs(sample) < 1e-6 and sample != 0:
        return "float32"
    return "float32"


def migrate_v1_to_v2(v1_data: dict) -> dict:
    """Convert v1 checkpoint to v2 format."""
    metadata = v1_data.get("metadata", {})
    v1_centers = v1_data.get("cluster_centers", {})

    if isinstance(v1_centers, dict):
        cluster_centers = v1_centers.get("cluster_centers", [])
        n_clusters = int(v1_centers.get("n_clusters", len(cluster_centers)))
    else:
        cluster_centers = v1_centers
        n_clusters = int(metadata.get("n_clusters", len(cluster_centers)))

    if not isinstance(cluster_centers, list) or not cluster_centers:
        raise ValueError(
            "Invalid v1 checkpoint: cluster_centers must be a non-empty 2D array"
        )
    if not all(isinstance(row, list) and row for row in cluster_centers):
        raise ValueError(
            "Invalid v1 checkpoint: cluster_centers must be a non-empty 2D array"
        )
    feature_dim = len(cluster_centers[0])
    if any(len(row) != feature_dim for row in cluster_centers):
        raise ValueError(
            "Invalid v1 checkpoint: cluster_centers rows must have equal length"
        )
    if n_clusters <= 0:
        raise ValueError(
            f"Invalid v1 checkpoint: n_clusters must be positive, got {n_clusters}"
        )
    if n_clusters != len(cluster_centers):
        raise ValueError(
            f"Invalid v1 checkpoint: n_clusters ({n_clusters}) does not match "
            f"len(cluster_centers) ({len(cluster_centers)})"
        )

    models_v2 = []
    for idx, m in enumerate(v1_data.get("models", [])):
        provider = m.get("provider", "").strip()
        model_name = m.get("model_name", "").strip()

        # Build model_id - validate we don't get empty or "/"
        if m.get("model_id"):
            model_id = m["model_id"].strip()
        elif provider and model_name:
            model_id = f"{provider}/{model_name}"
        elif provider:
            model_id = provider
        elif model_name:
            model_id = model_name
        else:
            print(
                f"Warning: Skipping model {idx} with empty provider and model_name",
                file=sys.stderr,
            )
            continue

        if not model_id or model_id == "/":
            print(
                f"Warning: Skipping model {idx} with invalid model_id: '{model_id}'",
                file=sys.stderr,
            )
            continue

        models_v2.append(
            {
                "model_id": model_id,
                "cost_per_1m_input_tokens": m.get("cost_per_1m_input_tokens", 0.0),
                "cost_per_1m_output_tokens": m.get("cost_per_1m_output_tokens", 0.0),
                "error_rates": m.get("error_rates", []),
            }
        )

    dtype = metadata.get("dtype", detect_dtype(cluster_centers))

    v2_data = {
        "version": "2.0",
        "cluster_centers": cluster_centers,
        "models": models_v2,
        "embedding": {
            "model": metadata.get("embedding_model", ""),
            "dtype": dtype,
            "trust_remote_code": metadata.get("allow_trust_remote_code", False),
        },
        "clustering": {
            "n_clusters": n_clusters,
            "random_state": metadata.get("random_state", 42),
            "max_iter": metadata.get("max_iter", 300),
            "n_init": metadata.get("n_init", 10),
            "algorithm": metadata.get("algorithm", "lloyd"),
            "normalization": metadata.get("normalization", "l2"),
        },
        "metrics": {
            "n_samples": metadata.get("n_train_questions", metadata.get("n_samples")),
            "cluster_sizes": metadata.get("cluster_sizes"),
            "silhouette_score": metadata.get("silhouette_score"),
            "inertia": metadata.get("inertia"),
        },
    }

    metrics = v2_data["metrics"]
    v2_data["metrics"] = {k: v for k, v in metrics.items() if v is not None}

    return v2_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate checkpoint from v1 to v2 format"
    )
    parser.add_argument("input", type=Path, help="Input v1 checkpoint (.json)")
    parser.add_argument(
        "output", type=Path, nargs="?", help="Output v2 checkpoint (.json)"
    )
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing")
    parser.add_argument("--compact", action="store_true", help="Compact JSON output")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    output_path = args.output or args.input.with_stem(args.input.stem + "_v2")

    if output_path.exists() and not args.force:
        print(
            f"Error: Output exists: {output_path} (use -f to overwrite)",
            file=sys.stderr,
        )
        return 1

    try:
        with open(args.input) as f:
            v1_data = json.load(f)

        if v1_data.get("version") == "2.0":
            print("Checkpoint is already v2 format", file=sys.stderr)
            return 1

        v2_data = migrate_v1_to_v2(v1_data)

        with open(output_path, "w") as f:
            if args.compact:
                json.dump(v2_data, f)
            else:
                json.dump(v2_data, f, indent=2)

        input_size = args.input.stat().st_size
        output_size = output_path.stat().st_size

        print(f"Migrated: {args.input} -> {output_path}")
        print("  Version: 1.0 -> 2.0")
        print(f"  Clusters: {v2_data['clustering']['n_clusters']}")
        print(f"  Models: {len(v2_data['models'])}")
        print(f"  Embedding: {v2_data['embedding']['model']}")
        print(f"  Input size: {input_size:,} bytes")
        print(f"  Output size: {output_size:,} bytes")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
