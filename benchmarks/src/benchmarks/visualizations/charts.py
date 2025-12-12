"""Chart generation for benchmark visualizations."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def generate_all_charts(
    results: dict[str, Any],
    output_dir: str | Path
) -> None:
    """Generate all visualization charts.

    Args:
        results: Benchmark results dict
        output_dir: Directory to save charts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = results["metrics"]

    # Generate static charts (Matplotlib/Seaborn)
    plot_model_distribution(metrics, output_dir)
    plot_agreement_by_category(metrics, output_dir)
    plot_latency_comparison(metrics, results["results"], output_dir)
    plot_error_vs_size(results["results"], output_dir)
    plot_routing_confusion(results["results"], output_dir)

    # Generate interactive dashboard (Plotly)
    generate_dashboard(results, output_dir)

    print(f"Charts saved to {output_dir}")


def plot_model_distribution(
    metrics: dict[str, Any],
    output_dir: Path
) -> None:
    """Chart 1: Side-by-side model distribution pie charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Profile router distribution
    profile_dist = metrics["routing"]["profile_router"]["model_distribution"]
    profile_models = list(profile_dist.keys())
    profile_counts = [v["count"] for v in profile_dist.values()]

    ax1.pie(
        profile_counts,
        labels=profile_models,
        autopct="%1.1f%%",
        startangle=90
    )
    ax1.set_title("Profile Router - Model Distribution", fontsize=16, fontweight="bold")

    # Claude router distribution
    claude_dist = metrics["routing"]["claude_router"]["model_distribution"]
    claude_models = list(claude_dist.keys())
    claude_counts = [v["count"] for v in claude_dist.values()]

    ax2.pie(
        claude_counts,
        labels=claude_models,
        autopct="%1.1f%%",
        startangle=90
    )
    ax2.set_title("Claude Oracle - Model Distribution", fontsize=16, fontweight="bold")

    plt.suptitle("Model Selection Distribution Comparison", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "1_model_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_agreement_by_category(
    metrics: dict[str, Any],
    output_dir: Path
) -> None:
    """Chart 2: Agreement rate by category bar chart."""
    by_cat = metrics["by_category"]

    categories = list(by_cat.keys())
    agreement_rates = [v["agreement_rate"] * 100 for v in by_cat.values()]
    counts = [v["count"] for v in by_cat.values()]

    # Sort by agreement rate
    sorted_indices = np.argsort(agreement_rates)[::-1]
    categories = [categories[i] for i in sorted_indices]
    agreement_rates = [agreement_rates[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    plt.figure(figsize=(14, 8))
    bars = plt.barh(categories, agreement_rates, color=sns.color_palette("viridis", len(categories)))

    # Add count labels
    for i, (rate, count) in enumerate(zip(agreement_rates, counts)):
        plt.text(rate + 1, i, f"{count} prompts", va="center")

    plt.xlabel("Agreement Rate (%)", fontsize=14)
    plt.ylabel("Category", fontsize=14)
    plt.title("Router Agreement Rate by Prompt Category", fontsize=16, fontweight="bold")
    plt.xlim(0, 105)
    plt.axvline(x=50, color="red", linestyle="--", alpha=0.5, label="50% Agreement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "2_agreement_by_category.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_latency_comparison(
    metrics: dict[str, Any],
    results: list[dict[str, Any]],
    output_dir: Path
) -> None:
    """Chart 3: Latency comparison box plot."""
    profile_latencies = [r["profile_router"]["e2e_latency_ms"] for r in results]
    claude_latencies = [r["claude_router"]["e2e_latency_ms"] for r in results]

    data = {
        "Profile Router": profile_latencies,
        "Claude Oracle": claude_latencies
    }

    plt.figure(figsize=(12, 8))
    box_plot = plt.boxplot(
        data.values(),
        labels=data.keys(),
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 8}
    )

    # Color boxes
    colors = ["#3498db", "#e74c3c"]
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel("End-to-End Latency (ms)", fontsize=14)
    plt.title("Inference Latency Comparison", fontsize=16, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)

    # Add stats text
    stats_text = f"Profile: μ={np.mean(profile_latencies):.1f}ms, P95={np.percentile(profile_latencies, 95):.1f}ms\n"
    stats_text += f"Claude: μ={np.mean(claude_latencies):.1f}ms, P95={np.percentile(claude_latencies, 95):.1f}ms"
    plt.text(
        0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_dir / "3_latency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_vs_size(
    results: list[dict[str, Any]],
    output_dir: Path
) -> None:
    """Chart 4: Error rate vs model size scatter plot."""
    # Aggregate by model
    model_stats = {}
    for r in results:
        for router_key in ["profile_router", "claude_router"]:
            router = r[router_key]
            model = router["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "size_mb": router["model_size_mb"],
                    "error_rates": [],
                    "router": router_key
                }
            model_stats[model]["error_rates"].append(router["error_rate"])

    # Calculate averages
    models = []
    sizes = []
    errors = []
    routers = []

    for model, stats in model_stats.items():
        models.append(model)
        sizes.append(stats["size_mb"])
        errors.append(np.mean(stats["error_rates"]) * 100)
        routers.append("Profile" if "profile" in stats["router"] else "Claude")

    plt.figure(figsize=(12, 8))

    # Plot points with different colors for each router
    for router_type, color, marker in [("Profile", "#3498db", "o"), ("Claude", "#e74c3c", "s")]:
        mask = [r == router_type for r in routers]
        plt.scatter(
            [s for s, m in zip(sizes, mask) if m],
            [e for e, m in zip(errors, mask) if m],
            s=150,
            alpha=0.6,
            c=color,
            marker=marker,
            label=router_type,
            edgecolors="black",
            linewidth=1
        )

    # Add trend line
    z = np.polyfit(sizes, errors, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(sizes), max(sizes), 100)
    plt.plot(x_trend, p(x_trend), "k--", alpha=0.3, label="Trend")

    plt.xlabel("Model Size (MB)", fontsize=14)
    plt.ylabel("Average Error Rate (%)", fontsize=14)
    plt.title("Model Quality vs Size Tradeoff", fontsize=16, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "4_error_vs_size.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_routing_confusion(
    results: list[dict[str, Any]],
    output_dir: Path
) -> None:
    """Chart 5: Routing confusion matrix heatmap."""
    # Extract model choices
    profile_choices = [r["profile_router"]["model"] for r in results]
    claude_choices = [r["claude_router"]["model"] for r in results]

    # Get unique models
    all_models = sorted(set(profile_choices + claude_choices))

    # Build confusion matrix
    conf_matrix = np.zeros((len(all_models), len(all_models)))
    model_to_idx = {model: i for i, model in enumerate(all_models)}

    for p_choice, c_choice in zip(profile_choices, claude_choices):
        i = model_to_idx[p_choice]
        j = model_to_idx[c_choice]
        conf_matrix[i, j] += 1

    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        conf_matrix,
        xticklabels=all_models,
        yticklabels=all_models,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Count"}
    )
    plt.xlabel("Claude Router Choice", fontsize=14)
    plt.ylabel("Profile Router Choice", fontsize=14)
    plt.title("Routing Decision Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "5_routing_confusion.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_dashboard(
    results: dict[str, Any],
    output_dir: Path
) -> None:
    """Generate interactive Plotly dashboard."""
    metrics = results["metrics"]

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Model Distribution (Profile)",
            "Model Distribution (Claude)",
            "Latency Comparison",
            "Agreement by Category",
            "Error Rate vs Model Size",
            "Performance Metrics Summary"
        ),
        specs=[
            [{"type": "pie"}, {"type": "pie"}],
            [{"type": "box"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "table"}]
        ]
    )

    # Chart 1 & 2: Model distributions
    profile_dist = metrics["routing"]["profile_router"]["model_distribution"]
    fig.add_trace(
        go.Pie(
            labels=list(profile_dist.keys()),
            values=[v["count"] for v in profile_dist.values()],
            name="Profile"
        ),
        row=1, col=1
    )

    claude_dist = metrics["routing"]["claude_router"]["model_distribution"]
    fig.add_trace(
        go.Pie(
            labels=list(claude_dist.keys()),
            values=[v["count"] for v in claude_dist.values()],
            name="Claude"
        ),
        row=1, col=2
    )

    # Chart 3: Latency box plot
    profile_latencies = [r["profile_router"]["e2e_latency_ms"] for r in results["results"]]
    claude_latencies = [r["claude_router"]["e2e_latency_ms"] for r in results["results"]]

    fig.add_trace(
        go.Box(y=profile_latencies, name="Profile Router", marker_color="#3498db"),
        row=2, col=1
    )
    fig.add_trace(
        go.Box(y=claude_latencies, name="Claude Oracle", marker_color="#e74c3c"),
        row=2, col=1
    )

    # Chart 4: Agreement by category
    by_cat = metrics["by_category"]
    fig.add_trace(
        go.Bar(
            x=[v["agreement_rate"] * 100 for v in by_cat.values()],
            y=list(by_cat.keys()),
            orientation="h",
            marker_color="viridis"
        ),
        row=2, col=2
    )

    # Chart 5: Error vs size scatter
    model_stats = {}
    for r in results["results"]:
        for router in [r["profile_router"], r["claude_router"]]:
            model = router["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "size": router["model_size_mb"],
                    "errors": []
                }
            model_stats[model]["errors"].append(router["error_rate"] * 100)

    sizes = [v["size"] for v in model_stats.values()]
    avg_errors = [np.mean(v["errors"]) for v in model_stats.values()]

    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=avg_errors,
            mode="markers",
            marker=dict(size=12, color=sizes, colorscale="Viridis", showscale=True),
            text=list(model_stats.keys()),
            hovertemplate="<b>%{text}</b><br>Size: %{x}MB<br>Error: %{y:.2f}%<extra></extra>"
        ),
        row=3, col=1
    )

    # Chart 6: Summary table
    perf = metrics["performance"]
    qual = metrics["quality"]

    summary_data = {
        "Metric": [
            "Avg Latency (ms)",
            "P95 Latency (ms)",
            "Avg Error Rate (%)",
            "Avg Memory (MB)"
        ],
        "Profile Router": [
            f"{perf['profile_router']['avg_latency_ms']:.1f}",
            f"{perf['profile_router']['p95_latency_ms']:.1f}",
            f"{qual['profile_router']['avg_error_rate']*100:.2f}",
            f"{perf['profile_router']['avg_memory_mb']:.0f}"
        ],
        "Claude Oracle": [
            f"{perf['claude_router']['avg_latency_ms']:.1f}",
            f"{perf['claude_router']['p95_latency_ms']:.1f}",
            f"{qual['claude_router']['avg_error_rate']*100:.2f}",
            f"{perf['claude_router']['avg_memory_mb']:.0f}"
        ]
    }

    fig.add_trace(
        go.Table(
            header=dict(
                values=list(summary_data.keys()),
                fill_color="paleturquoise",
                align="left"
            ),
            cells=dict(
                values=list(summary_data.values()),
                fill_color="lavender",
                align="left"
            )
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Cactus Router vs Claude Oracle - Interactive Dashboard",
        title_font_size=20,
        showlegend=True,
        height=1400
    )

    # Save
    fig.write_html(output_dir / "dashboard.html")
    print(f"Dashboard saved to {output_dir / 'dashboard.html'}")
