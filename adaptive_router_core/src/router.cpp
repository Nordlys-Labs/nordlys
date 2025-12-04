#include "router.hpp"

void Router::initialize(const RouterProfile& prof) {
    profile_ = prof;
    embedding_dim_ = static_cast<int>(prof.cluster_centers.cols());

    cluster_engine_.load_centroids(prof.cluster_centers);

    scorer_.load_models(prof.models);
    scorer_.set_lambda_params(
        prof.metadata.routing.lambda_min,
        prof.metadata.routing.lambda_max
    );
}

Router Router::from_file(const std::string& profile_path) {
    Router router;
    auto profile = RouterProfile::from_json(profile_path);
    router.initialize(profile);
    return router;
}

Router Router::from_json_string(const std::string& json_str) {
    Router router;
    auto profile = RouterProfile::from_json_string(json_str);
    router.initialize(profile);
    return router;
}

Router Router::from_binary(const std::string& path) {
    Router router;
    auto profile = RouterProfile::from_binary(path);
    router.initialize(profile);
    return router;
}

RouteResponse Router::route(const RouteRequest& request) {
    // Convert span to Eigen vector
    EmbeddingVector embedding(static_cast<Eigen::Index>(request.embedding.size()));
    for (size_t i = 0; i < request.embedding.size(); ++i) {
        embedding(static_cast<Eigen::Index>(i)) = request.embedding[i];
    }

    return route(embedding, request.cost_bias);
}

RouteResponse Router::route(
    const float* embedding_data,
    size_t embedding_size,
    float cost_bias
) {
    EmbeddingVector embedding(static_cast<Eigen::Index>(embedding_size));
    for (size_t i = 0; i < embedding_size; ++i) {
        embedding(static_cast<Eigen::Index>(i)) = embedding_data[i];
    }

    return route(embedding, cost_bias);
}

RouteResponse Router::route(
    const EmbeddingVector& embedding,
    float cost_bias
) {
    // Assign to cluster
    auto [cluster_id, distance] = cluster_engine_.assign(embedding);

    // Score models for this cluster
    auto scores = scorer_.score_models(cluster_id, cost_bias);

    RouteResponse response;
    response.cluster_id = cluster_id;
    response.cluster_distance = distance;

    if (!scores.empty()) {
        response.selected_model = scores[0].model_id;

        // Add alternatives (skip first, which is selected)
        for (size_t i = 1; i < scores.size() && i <= 5; ++i) {
            response.alternatives.push_back(scores[i].model_id);
        }
    }

    return response;
}

std::vector<std::string> Router::get_supported_models() const {
    std::vector<std::string> models;
    for (const auto& m : profile_.models) {
        models.push_back(m.model_id);
    }
    return models;
}

int Router::get_n_clusters() const {
    return cluster_engine_.get_n_clusters();
}

int Router::get_embedding_dim() const {
    return embedding_dim_;
}
