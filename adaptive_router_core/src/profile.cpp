#include "profile.hpp"
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <msgpack.hpp>

RouterProfile RouterProfile::from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open profile file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_json_string(buffer.str());
}

RouterProfile RouterProfile::from_json_string(const std::string& json_str) {
    auto json = nlohmann::json::parse(json_str);
    RouterProfile profile;

    // Parse cluster centers
    auto& centers_json = json["cluster_centers"];
    int n_clusters = centers_json["n_clusters"];
    int feature_dim = centers_json["feature_dim"];
    auto& centers_data = centers_json["cluster_centers"];

    profile.cluster_centers.resize(n_clusters, feature_dim);
    for (int i = 0; i < n_clusters; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            profile.cluster_centers(i, j) = centers_data[i][j].get<float>();
        }
    }

    // Parse models
    for (const auto& m : json["models"]) {
        ModelFeatures model;
        model.provider = m["provider"];
        model.model_name = m["model_name"];
        model.model_id = model.provider + "/" + model.model_name;
        model.cost_per_1m_input_tokens = m["cost_per_1m_input_tokens"];
        model.cost_per_1m_output_tokens = m["cost_per_1m_output_tokens"];

        for (const auto& er : m["error_rates"]) {
            model.error_rates.push_back(er.get<float>());
        }

        profile.models.push_back(std::move(model));
    }

    // Parse metadata
    auto& meta = json["metadata"];
    profile.metadata.n_clusters = meta["n_clusters"];
    profile.metadata.embedding_model = meta["embedding_model"];
    profile.metadata.silhouette_score = meta.value("silhouette_score", 0.0f);

    if (meta.contains("clustering")) {
        auto& cl = meta["clustering"];
        profile.metadata.clustering.max_iter = cl.value("max_iter", 300);
        profile.metadata.clustering.random_state = cl.value("random_state", 42);
        profile.metadata.clustering.n_init = cl.value("n_init", 10);
        profile.metadata.clustering.algorithm = cl.value("algorithm", "lloyd");
        profile.metadata.clustering.normalization_strategy = cl.value("normalization_strategy", "l2");
    }

    if (meta.contains("routing")) {
        auto& rt = meta["routing"];
        profile.metadata.routing.lambda_min = rt.value("lambda_min", 0.0f);
        profile.metadata.routing.lambda_max = rt.value("lambda_max", 2.0f);
        profile.metadata.routing.default_cost_preference = rt.value("default_cost_preference", 0.5f);
    }

    return profile;
}

RouterProfile RouterProfile::from_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary profile file: " + path);
    }

    std::string buffer((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

    auto handle = msgpack::unpack(buffer.data(), buffer.size());
    auto obj = handle.get();
    auto map = obj.as<std::map<std::string, msgpack::object>>();

    RouterProfile profile;

    // Parse cluster centers from raw bytes
    auto centers_map = map["cluster_centers"].as<std::map<std::string, msgpack::object>>();
    int n_clusters = centers_map["n_clusters"].as<int>();
    int feature_dim = centers_map["feature_dim"].as<int>();
    std::string centers_bytes = centers_map["data"].as<std::string>();

    profile.cluster_centers.resize(n_clusters, feature_dim);
    std::memcpy(profile.cluster_centers.data(), centers_bytes.data(),
                n_clusters * feature_dim * sizeof(float));

    // Parse models
    auto models_arr = map["models"].as<std::vector<msgpack::object>>();
    for (const auto& m_obj : models_arr) {
        auto m = m_obj.as<std::map<std::string, msgpack::object>>();
        ModelFeatures model;
        model.provider = m["provider"].as<std::string>();
        model.model_name = m["model_name"].as<std::string>();
        model.model_id = model.provider + "/" + model.model_name;
        model.cost_per_1m_input_tokens = m["cost_per_1m_input_tokens"].as<float>();
        model.cost_per_1m_output_tokens = m["cost_per_1m_output_tokens"].as<float>();
        model.error_rates = m["error_rates"].as<std::vector<float>>();
        profile.models.push_back(std::move(model));
    }

    // Parse metadata
    auto meta = map["metadata"].as<std::map<std::string, msgpack::object>>();
    profile.metadata.n_clusters = meta["n_clusters"].as<int>();
    profile.metadata.embedding_model = meta["embedding_model"].as<std::string>();

    if (meta.count("silhouette_score")) {
        profile.metadata.silhouette_score = meta["silhouette_score"].as<float>();
    }

    return profile;
}
