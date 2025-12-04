#include <gtest/gtest.h>
#include "cluster.hpp"

TEST(ClusterEngineTest, EmptyEngine) {
    ClusterEngine engine;
    EXPECT_EQ(engine.get_n_clusters(), 0);
}

TEST(ClusterEngineTest, LoadCentroids) {
    ClusterEngine engine;

    // Create simple 3 clusters with 4D embeddings
    EmbeddingMatrix centers(3, 4);
    centers << 1.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 1.0f, 0.0f, 0.0f,
               0.0f, 0.0f, 1.0f, 0.0f;

    engine.load_centroids(centers);
    EXPECT_EQ(engine.get_n_clusters(), 3);
}

TEST(ClusterEngineTest, AssignToNearestCluster) {
    ClusterEngine engine;

    EmbeddingMatrix centers(3, 4);
    centers << 1.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 1.0f, 0.0f, 0.0f,
               0.0f, 0.0f, 1.0f, 0.0f;

    engine.load_centroids(centers);

    // Test vector close to first centroid
    EmbeddingVector vec(4);
    vec << 0.9f, 0.1f, 0.0f, 0.0f;

    auto [cluster_id, distance] = engine.assign(vec);
    EXPECT_EQ(cluster_id, 0);
    EXPECT_GT(distance, 0.0f);
}
