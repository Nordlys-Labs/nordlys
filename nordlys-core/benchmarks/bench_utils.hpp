#pragma once

#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace bench_utils {

  inline std::string GetFixturePath(const std::string& filename) {
    // Try multiple possible locations for fixtures
    std::vector<std::filesystem::path> search_paths
        = {// When running from build directory (cmake copies fixtures there)
           std::filesystem::current_path() / "fixtures",
           // When running from nordlys-core directory
           std::filesystem::current_path() / "benchmarks" / "fixtures",
           // Absolute path to source directory
           std::filesystem::path(__FILE__).parent_path() / "fixtures"};

    for (const auto& base_path : search_paths) {
      auto full_path = base_path / filename;
      if (std::filesystem::exists(full_path)) {
        return full_path.string();
      }
    }

    // Fallback - return first path (will error with clear message)
    return (search_paths[0] / filename).string();
  }

  inline std::vector<float> GenerateRandomEmbedding(size_t dim, uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> embedding(dim);
    for (size_t i = 0; i < dim; ++i) {
      embedding[i] = dist(gen);
    }

    return embedding;
  }

  inline std::vector<std::vector<float>> GenerateBatchEmbeddings(size_t batch_size, size_t dim,
                                                                 uint32_t seed = 42) {
    std::vector<std::vector<float>> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
      batch.push_back(GenerateRandomEmbedding(dim, seed + i));
    }

    return batch;
  }

}  // namespace bench_utils
