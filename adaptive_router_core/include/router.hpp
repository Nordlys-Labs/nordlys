#pragma once
#include <string>
#include <vector>
#include <span>
#include "types.hpp"
#include "scorer.hpp"
#include "cluster.hpp"
#include "profile.hpp"

struct RouteRequest {
    // Pre-computed embedding vector (from Python sentence-transformers)
    std::span<const float> embedding;
    float cost_bias = 0.5f;
    std::vector<std::string> models;  // Optional filter
};

struct RouteResponse {
    std::string selected_model;
    std::vector<std::string> alternatives;
    int cluster_id;
    float cluster_distance;
};

class Router {
public:
    // Factory methods
    [[nodiscard]] static Router from_file(const std::string& profile_path);
    [[nodiscard]] static Router from_json_string(const std::string& json_str);
    [[nodiscard]] static Router from_binary(const std::string& path);

    // Default constructors/destructor
    Router() = default;
    ~Router() = default;

    // Movable
    Router(Router&&) = default;
    Router& operator=(Router&&) = default;
    Router(const Router&) = delete;
    Router& operator=(const Router&) = delete;

    // Main routing API - accepts pre-computed embedding
    [[nodiscard]] RouteResponse route(const RouteRequest& request);

    // Convenience overload with raw float pointer
    [[nodiscard]] RouteResponse route(
        const float* embedding_data,
        size_t embedding_size,
        float cost_bias = 0.5f
    );

    // Convenience overload with Eigen vector
    [[nodiscard]] RouteResponse route(
        const EmbeddingVector& embedding,
        float cost_bias = 0.5f
    );

    // Introspection
    [[nodiscard]] std::vector<std::string> get_supported_models() const;
    [[nodiscard]] int get_n_clusters() const;
    [[nodiscard]] int get_embedding_dim() const;

private:
    void initialize(const RouterProfile& prof);

    ClusterEngine cluster_engine_;
    ModelScorer scorer_;
    RouterProfile profile_;
    int embedding_dim_ = 384;  // Default for all-MiniLM-L6-v2
};
