#!/usr/bin/env python3
"""
Model Profiling via SWE-bench Results Mapping

Maps existing SWE-bench evaluation results (from GitHub) to clusters.
Supports both 'verified' and 'bash-only' evaluation types.

Usage:
    # Profile a verified model (default)
    python profile_model.py --model-folder "20240620_sweagent_claude3.5sonnet"

    # Profile a bash-only model
    python profile_model.py --model-folder "20251124_mini-v1.16.0_claude-opus-4-5" \\
        --eval-type bash-only

    # List available models
    python profile_model.py --list-models
    python profile_model.py --list-models --eval-type bash-only

    # Add a single model from profiles/ to profile.json (without re-reading all)
    python profile_model.py --add "20250929_mini-v1.13.3_sonnet-4-5-20250929"

    # Combine all profiles into final profile.json
    python profile_model.py --combine --output profile.json
"""

import argparse
import base64
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from github import Auth, Github

# Load .env file from clustering directory
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

REPO_NAME = "swe-bench/experiments"

# Evaluation type configurations
EVAL_CONFIGS = {
    "verified": {
        "path": "evaluation/verified",
        "results_file": "results/results.json",
    },
    "bash-only": {
        "path": "evaluation/bash-only",
        "results_file": "per_instance_details.json",
    },
}

# Model cost registry - costs are fetched from external registry at runtime.
# Individual profiles use default costs (1.0) as placeholders.
MODEL_COSTS: dict[str, dict[str, float]] = {}


def get_github_client() -> Github:
    """Create GitHub client, using token if available for higher rate limits."""
    token = os.environ.get("GITHUB_TOKEN")
    if token and token.strip():
        print("  Using GitHub token for authentication")
        auth = Auth.Token(token.strip())
        return Github(auth=auth)
    print("  No GitHub token found, using unauthenticated access")
    return Github()


def load_verified_assignments(clusters_dir: Path) -> dict[str, int]:
    """Load verified instance → cluster mappings."""
    assignments_file = clusters_dir / "verified_cluster_assignments.json"
    with open(assignments_file) as f:
        data = json.load(f)
    return dict(zip(data["instance_ids"], data["cluster_labels"]))


def load_cluster_centers(clusters_dir: Path) -> dict:
    """Load cluster centers for final profile."""
    centroids_file = clusters_dir / "selected_cluster_centroids.json"
    with open(centroids_file) as f:
        data = json.load(f)

    centers_dict = data["cluster_centers_768d"]
    n_clusters = len(centers_dict)

    return {
        "n_clusters": n_clusters,
        "feature_dim": 768,
        "cluster_centers": [centers_dict[str(i)] for i in range(n_clusters)],
    }


def load_metadata(clusters_dir: Path) -> dict:
    """Load metadata from centroids file."""
    centroids_file = clusters_dir / "selected_cluster_centroids.json"
    with open(centroids_file) as f:
        data = json.load(f)

    return {
        "n_clusters": data["metrics"]["n_clusters"],
        "silhouette_score": data["metrics"]["silhouette"],
        "embedding_model": data["metadata"]["embedding_model"],
        "embedding_dim": data["metadata"]["embedding_dim"],
    }


def parse_verified_results(data: dict) -> set[str]:
    """Parse verified format: {"resolved": ["id1", ...]}"""
    return set(data.get("resolved", []))


def parse_bashonly_results(data: dict) -> set[str]:
    """Parse bash-only format: {"id": {"resolved": true/false, ...}}"""
    return {id for id, info in data.items() if info.get("resolved", False)}


def fetch_github_results(gh: Github, model_folder: str, eval_type: str) -> set[str]:
    """Fetch results from GitHub and return resolved instance IDs."""
    config = EVAL_CONFIGS[eval_type]
    repo = gh.get_repo(REPO_NAME)
    file_path = f"{config['path']}/{model_folder}/{config['results_file']}"

    content = repo.get_contents(file_path)
    if isinstance(content, list):
        content = content[0]
    decoded = base64.b64decode(content.content).decode("utf-8")
    data = json.loads(decoded)

    if eval_type == "verified":
        return parse_verified_results(data)
    else:  # bash-only
        return parse_bashonly_results(data)


def list_available_models(gh: Github, eval_type: str) -> list[str]:
    """List all available model folders from GitHub."""
    config = EVAL_CONFIGS[eval_type]
    repo = gh.get_repo(REPO_NAME)
    result = repo.get_contents(config["path"])
    if isinstance(result, list):
        contents = result
    else:
        contents = [result]
    return sorted([item.name for item in contents if item.type == "dir"])


def parse_model_info(model_folder: str) -> tuple[str, str]:
    """Extract provider and model name from folder name.

    Model name is the last part after "_" in the folder name.
    Provider is inferred from the model name.
    """
    # Extract model_name as last part after "_"
    model_name = model_folder.split("_")[-1]
    model_lower = model_name.lower()

    # Detect provider from model name
    if "claude" in model_lower:
        provider = "anthropic"
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        provider = "openai"
    elif "deepseek" in model_lower:
        provider = "deepseek"
    elif "qwen" in model_lower:
        provider = "qwen"
    elif "gemini" in model_lower:
        provider = "google"
    elif "llama" in model_lower or "codellama" in model_lower:
        provider = "meta"
    elif "mistral" in model_lower or "mixtral" in model_lower:
        provider = "mistral"
    else:
        provider = "unknown"

    return provider, model_name


def calculate_cluster_stats(
    assignments: dict[str, int], resolved: set[str]
) -> dict[int, dict]:
    """Calculate per-cluster statistics."""
    cluster_stats = {}

    for instance_id, cluster in assignments.items():
        if cluster not in cluster_stats:
            cluster_stats[cluster] = {"total": 0, "resolved": 0}
        cluster_stats[cluster]["total"] += 1
        if instance_id in resolved:
            cluster_stats[cluster]["resolved"] += 1

    return cluster_stats


def profile_single_model(
    gh: Github,
    model_folder: str,
    clusters_dir: Path,
    output_dir: Path,
    eval_type: str,
) -> dict:
    """Profile a single model and save to profiles/{model_folder}/profile.json."""
    print(f"Profiling {model_folder} ({eval_type})...")

    # Load verified assignments (same for both eval types)
    assignments = load_verified_assignments(clusters_dir)
    n_clusters = max(assignments.values()) + 1

    # Fetch results from GitHub
    resolved = fetch_github_results(gh, model_folder, eval_type)
    print(f"  Found {len(resolved)} resolved instances")

    # Calculate stats
    cluster_stats = calculate_cluster_stats(assignments, resolved)

    # Build error rates list (ordered by cluster index)
    error_rates = []
    for i in range(n_clusters):
        if i in cluster_stats and cluster_stats[i]["total"] > 0:
            rate = 1 - (cluster_stats[i]["resolved"] / cluster_stats[i]["total"])
            error_rates.append(round(rate, 4))
        else:
            error_rates.append(1.0)

    # Parse model info
    provider, model_name = parse_model_info(model_folder)

    total_evaluated = sum(s["total"] for s in cluster_stats.values())
    total_resolved = sum(s["resolved"] for s in cluster_stats.values())

    profile = {
        "model_folder": model_folder,
        "eval_type": eval_type,
        "provider": provider,
        "model_name": model_name,
        "error_rates": error_rates,
        "cluster_stats": {str(k): v for k, v in sorted(cluster_stats.items())},
        "overall_error_rate": round(1 - (total_resolved / total_evaluated), 4),
        "total_evaluated": total_evaluated,
        "total_resolved": total_resolved,
        "created_at": datetime.now().isoformat(),
    }

    # Save
    model_dir = output_dir / model_folder
    model_dir.mkdir(parents=True, exist_ok=True)
    output_file = model_dir / "profile.json"
    with open(output_file, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"  Resolved: {total_resolved}/{total_evaluated}")
    print(f"  Error rates: {error_rates}")
    print(f"  Saved to: {output_file}")

    return profile


def add_model_to_profile(
    model_folder: str, profiles_dir: Path, profile_path: Path
) -> None:
    """Add a single model from profiles/ to the main profile.json.

    Args:
        model_folder: Model folder name (e.g., "20250929_mini-...")
        profiles_dir: Directory containing individual profiles
        profile_path: Path to main profile.json
    """
    # Find model profile
    model_profile_path = profiles_dir / model_folder / "profile.json"
    if not model_profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {model_profile_path}")

    # Load model profile
    with open(model_profile_path) as f:
        model_data = json.load(f)

    # Load main profile
    if not profile_path.exists():
        raise FileNotFoundError(f"Main profile not found: {profile_path}")

    with open(profile_path) as f:
        profile = json.load(f)

    # Build model entry
    model_name = model_data["model_name"]
    costs = MODEL_COSTS.get(
        model_name, {"input": 1.0, "output": 1.0, "provider": model_data["provider"]}
    )

    new_model = {
        "provider": model_data["provider"],
        "model_name": model_data["model_name"],
        "model_folder": model_data["model_folder"],
        "eval_type": model_data.get("eval_type", "verified"),
        "cost_per_1m_input_tokens": costs["input"],
        "cost_per_1m_output_tokens": costs["output"],
        "error_rates": model_data["error_rates"],
    }

    # Check if model already exists (by model_folder)
    models = profile.get("models", [])
    existing_idx = None
    for i, m in enumerate(models):
        if m.get("model_folder") == model_folder:
            existing_idx = i
            break

    if existing_idx is not None:
        print(f"Updating existing model: {model_folder}")
        models[existing_idx] = new_model
    else:
        print(f"Adding new model: {model_folder}")
        models.append(new_model)

    profile["models"] = models

    # Save
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Added {model_name} to {profile_path}")
    print(f"   Error rates: {new_model['error_rates']}")
    print(f"   Total models in profile: {len(models)}")


def combine_profiles(profiles_dir: Path, clusters_dir: Path, output_path: Path) -> None:
    """Combine all individual profiles into final profile.json."""
    print("Combining profiles...")

    # Load cluster centers
    cluster_centers = load_cluster_centers(clusters_dir)

    # Load metadata
    metadata = load_metadata(clusters_dir)

    # Load all individual profiles
    models = []
    for profile_file in sorted(profiles_dir.glob("*/profile.json")):
        with open(profile_file) as f:
            model = json.load(f)

        model_name = model["model_name"]
        costs = MODEL_COSTS.get(
            model_name, {"input": 1.0, "output": 1.0, "provider": model["provider"]}
        )

        models.append(
            {
                "provider": model["provider"],
                "model_name": model["model_name"],
                "model_folder": model["model_folder"],
                "eval_type": model.get("eval_type", "verified"),
                "cost_per_1m_input_tokens": costs["input"],
                "cost_per_1m_output_tokens": costs["output"],
                "error_rates": model["error_rates"],
            }
        )

    # Count verified instances
    assignments = load_verified_assignments(clusters_dir)
    metadata["n_verified_instances"] = len(assignments)

    # Build final profile
    final_profile = {
        "cluster_centers": cluster_centers,
        "models": models,
        "metadata": metadata,
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(final_profile, f, indent=2)

    print(f"Combined {len(models)} models into: {output_path}")


def main() -> None:
    """CLI entry point for model profiling operations."""
    parser = argparse.ArgumentParser(
        description="Generate model profiles from SWE-bench experiment results"
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        help="Model folder name from swe-bench/experiments",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["verified", "bash-only"],
        default="verified",
        help="Evaluation type: verified (default) or bash-only",
    )
    parser.add_argument(
        "--clusters-dir",
        type=Path,
        default=Path(__file__).parent / "clustering_21122025",
        help="Directory containing cluster data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "profiles",
        help="Output directory for individual profiles",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model folders",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all profiles into final profile.json",
    )
    parser.add_argument(
        "--add",
        type=str,
        metavar="MODEL_FOLDER",
        help="Add a single model from profiles/ to profile.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "profile.json",
        help="Output path for combined profile",
    )

    args = parser.parse_args()

    gh = get_github_client()

    if args.list_models:
        print(f"Fetching available {args.eval_type} models from GitHub...")
        models = list_available_models(gh, args.eval_type)
        print(f"\nAvailable {args.eval_type} models ({len(models)} total):\n")
        for model in models:
            print(f"  {model}")
        return

    if args.combine:
        combine_profiles(args.output_dir, args.clusters_dir, args.output)
        return

    if args.add:
        add_model_to_profile(args.add, args.output_dir, args.output)
        return

    if not args.model_folder:
        parser.error("--model-folder is required (or use --list-models / --combine)")

    profile_single_model(
        gh, args.model_folder, args.clusters_dir, args.output_dir, args.eval_type
    )


if __name__ == "__main__":
    main()
