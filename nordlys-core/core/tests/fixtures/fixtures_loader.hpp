#pragma once

#include <cmath>
#include <random>
#include <vector>

namespace nordlys::testing {

  class FixturesLoader {
  public:
    static std::vector<std::vector<float>> generate_embeddings(size_t n, size_t dim,
                                                               unsigned seed = 42) {
      std::mt19937 rng(seed);
      std::normal_distribution<float> dist(0.0f, 1.0f);

      std::vector<std::vector<float>> embeddings(n, std::vector<float>(dim));
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          embeddings[i][j] = dist(rng);
        }
        l2_normalize(embeddings[i]);
      }
      return embeddings;
    }

    static std::vector<std::vector<float>> generate_clustered_embeddings(size_t n, size_t dim,
                                                                         size_t n_clusters,
                                                                         unsigned seed = 42) {
      std::mt19937 rng(seed);
      std::normal_distribution<float> cluster_center_dist(0.0f, 1.0f);
      std::normal_distribution<float> noise_dist(0.0f, 0.1f);

      std::vector<std::vector<float>> cluster_centers(n_clusters, std::vector<float>(dim));
      for (size_t c = 0; c < n_clusters; ++c) {
        for (size_t d = 0; d < dim; ++d) {
          cluster_centers[c][d] = cluster_center_dist(rng);
        }
        l2_normalize(cluster_centers[c]);
      }

      std::vector<std::vector<float>> embeddings(n, std::vector<float>(dim));
      std::uniform_int_distribution<size_t> cluster_dist(0, n_clusters - 1);

      for (size_t i = 0; i < n; ++i) {
        size_t cluster_idx = cluster_dist(rng);
        for (size_t d = 0; d < dim; ++d) {
          embeddings[i][d] = cluster_centers[cluster_idx][d] + noise_dist(rng);
        }
        l2_normalize(embeddings[i]);
      }
      return embeddings;
    }

    static std::vector<std::vector<double>> generate_embeddings_double(size_t n, size_t dim,
                                                                       unsigned seed = 42) {
      std::mt19937 rng(seed);
      std::normal_distribution<double> dist(0.0, 1.0);

      std::vector<std::vector<double>> embeddings(n, std::vector<double>(dim));
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          embeddings[i][j] = dist(rng);
        }
        l2_normalize_double(embeddings[i]);
      }
      return embeddings;
    }

  private:
    static void l2_normalize(std::vector<float>& vec) {
      float norm = 0.0f;
      for (float val : vec) {
        norm += val * val;
      }
      norm = std::sqrt(norm);
      if (norm > 1e-10f) {
        for (float& val : vec) {
          val /= norm;
        }
      }
    }

    static void l2_normalize_double(std::vector<double>& vec) {
      double norm = 0.0;
      for (double val : vec) {
        norm += val * val;
      }
      norm = std::sqrt(norm);
      if (norm > 1e-10) {
        for (double& val : vec) {
          val /= norm;
        }
      }
    }
  };

}  // namespace nordlys::testing
