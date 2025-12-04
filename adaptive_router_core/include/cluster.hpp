#pragma once
#include <utility>
#include "types.hpp"

class ClusterEngine {
public:
    ClusterEngine() = default;
    ~ClusterEngine() = default;

    // Movable
    ClusterEngine(ClusterEngine&&) = default;
    ClusterEngine& operator=(ClusterEngine&&) = default;
    ClusterEngine(const ClusterEngine&) = delete;
    ClusterEngine& operator=(const ClusterEngine&) = delete;

    // Load K-means cluster centers (K x D matrix)
    void load_centroids(const EmbeddingMatrix& centers);

    // Assign embedding to nearest cluster
    // Returns (cluster_id, distance) pair
    [[nodiscard]] std::pair<int, float> assign(const EmbeddingVector& embedding);

    // Get number of clusters
    [[nodiscard]] int get_n_clusters() const;

private:
    EmbeddingMatrix centroids_;
    int n_clusters_ = 0;
};
