#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nordlys_core/checkpoint.hpp>

#include "bindings.h"

namespace nb = nanobind;

void register_types(nb::module_& m) {
  // TrainingMetrics type
  nb::class_<TrainingMetrics>(m, "TrainingMetrics", "Training metrics (optional fields)")
      .def_prop_ro(
          "n_samples", [](const TrainingMetrics& m) -> std::optional<int> { return m.n_samples; },
          "Number of training samples (None if not available)")
      .def_prop_ro(
          "cluster_sizes",
          [](const TrainingMetrics& m) -> std::optional<std::vector<int>> {
            return m.cluster_sizes;
          },
          "Cluster sizes (None if not available)")
      .def_prop_ro(
          "silhouette_score",
          [](const TrainingMetrics& m) -> std::optional<float> { return m.silhouette_score; },
          "Silhouette score (None if not available)")
      .def_prop_ro(
          "inertia", [](const TrainingMetrics& m) -> std::optional<float> { return m.inertia; },
          "Inertia (None if not available)");

  // EmbeddingConfig type
  nb::class_<EmbeddingConfig>(m, "EmbeddingConfig", "Embedding configuration")
      .def_ro("model", &EmbeddingConfig::model, "Embedding model ID")
      .def_ro("trust_remote_code", &EmbeddingConfig::trust_remote_code,
              "Whether to trust remote code");

  // ClusteringConfig type
  nb::class_<ClusteringConfig>(m, "ClusteringConfig", "Clustering configuration parameters")
      .def_ro("n_clusters", &ClusteringConfig::n_clusters, "Number of clusters")
      .def_ro("random_state", &ClusteringConfig::random_state, "Random state for reproducibility")
      .def_ro("max_iter", &ClusteringConfig::max_iter, "Maximum iterations")
      .def_ro("n_init", &ClusteringConfig::n_init, "Number of initializations")
      .def_ro("algorithm", &ClusteringConfig::algorithm, "Clustering algorithm")
      .def_ro("normalization", &ClusteringConfig::normalization, "Normalization strategy");

  // ModelFeatures type
  nb::class_<ModelFeatures>(m, "ModelFeatures", "Model configuration with error rates")
      .def_ro("model_id", &ModelFeatures::model_id, "Full model ID (e.g., 'openai/gpt-4')")
      .def_ro("error_rates", &ModelFeatures::error_rates, "Per-cluster error rates")
      .def_ro("cost_per_1m_input_tokens", &ModelFeatures::cost_per_1m_input_tokens,
              "Cost per 1M input tokens")
      .def_ro("cost_per_1m_output_tokens", &ModelFeatures::cost_per_1m_output_tokens,
              "Cost per 1M output tokens")
      .def("provider", &ModelFeatures::provider, "Extract provider from model_id")
      .def("model_name", &ModelFeatures::model_name, "Extract model name from model_id")
      .def("cost_per_1m_tokens", &ModelFeatures::cost_per_1m_tokens, "Average cost per 1M tokens");
}
