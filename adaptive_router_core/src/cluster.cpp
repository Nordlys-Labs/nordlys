#include "cluster.hpp"

void ClusterEngine::load_centroids(const EmbeddingMatrix& centers) {
    centroids_ = centers;
    n_clusters_ = static_cast<int>(centers.rows());
}

std::pair<int, float> ClusterEngine::assign(const EmbeddingVector& embedding) {
    if (n_clusters_ == 0) {
        return {-1, 0.0f};
    }

    // Compute squared Euclidean distances to all centroids
    int best_cluster = 0;
    float best_distance = std::numeric_limits<float>::max();

    for (int i = 0; i < n_clusters_; ++i) {
        float dist = (centroids_.row(i).transpose() - embedding).squaredNorm();
        if (dist < best_distance) {
            best_distance = dist;
            best_cluster = i;
        }
    }

    return {best_cluster, std::sqrt(best_distance)};
}

int ClusterEngine::get_n_clusters() const {
    return n_clusters_;
}
