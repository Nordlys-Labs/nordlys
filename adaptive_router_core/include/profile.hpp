#pragma once
#include <string>
#include <vector>
#include <memory>
#include "types.hpp"
#include "scorer.hpp"

struct ClusteringConfig {
    int max_iter = 300;
    int random_state = 42;
    int n_init = 10;
    std::string algorithm = "lloyd";
    std::string normalization_strategy = "l2";
};

struct RoutingConfig {
    float lambda_min = 0.0f;
    float lambda_max = 2.0f;
    float default_cost_preference = 0.5f;
};

struct ProfileMetadata {
    int n_clusters;
    std::string embedding_model;
    float silhouette_score;
    ClusteringConfig clustering;
    RoutingConfig routing;
};

struct RouterProfile {
    EmbeddingMatrix cluster_centers;  // K x D matrix
    std::vector<ModelFeatures> models;
    ProfileMetadata metadata;

    // Load from JSON file
    [[nodiscard]] static RouterProfile from_json(const std::string& path);

    // Load from MessagePack binary file
    [[nodiscard]] static RouterProfile from_binary(const std::string& path);

    // Load from JSON string
    [[nodiscard]] static RouterProfile from_json_string(const std::string& json_str);
};
