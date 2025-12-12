#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <ranges>
#include <span>

#include "adaptive.h"
#include <adaptive_core/router.hpp>

// Internal helper to convert std::string to C string
static char* str_duplicate(const std::string& str) {
  char* result = static_cast<char*>(malloc(str.length() + 1));
  if (result) {
    std::ranges::copy(str, result);
    result[str.length()] = '\0';
  }
  return result;
}

// Internal helper to clean up route result contents (but not the struct itself)
static void cleanup_route_result_contents(AdaptiveRouteResult* result) {
  if (!result) return;

  free(result->selected_model);
  result->selected_model = nullptr;

  if (result->alternatives) {
    // Use ranges to free alternatives
    auto alternatives_span = std::span(result->alternatives, result->alternatives_count);
    std::ranges::for_each(alternatives_span, [](char* str) { free(str); });
    free(result->alternatives);
    result->alternatives = nullptr;
  }
  result->alternatives_count = 0;
}

// C API implementation
extern "C" {

AdaptiveRouter* adaptive_router_create(const char* profile_path) {
  if (!profile_path) {
    return nullptr;
  }

  try {
    auto result = Router::from_file(profile_path);
    if (!result) {
      return nullptr;
    }
    return reinterpret_cast<AdaptiveRouter*>(new Router(std::move(*result)));
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

AdaptiveRouter* adaptive_router_create_from_json(const char* json_str) {
  if (!json_str) {
    return nullptr;
  }

  try {
    auto result = Router::from_json_string(json_str);
    if (!result) {
      return nullptr;
    }
    return reinterpret_cast<AdaptiveRouter*>(new Router(std::move(*result)));
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

AdaptiveRouter* adaptive_router_create_from_binary(const char* path) {
  if (!path) {
    return nullptr;
  }

  try {
    auto result = Router::from_binary(path);
    if (!result) {
      return nullptr;
    }
    return reinterpret_cast<AdaptiveRouter*>(new Router(std::move(*result)));
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

void adaptive_router_destroy(AdaptiveRouter* router) {
  if (router) {
    delete reinterpret_cast<Router*>(router);
  }
}

AdaptiveRouteResult* adaptive_router_route(AdaptiveRouter* router, const float* embedding,
                                           size_t embedding_size, float cost_bias) {
  if (!router || !embedding) {
    return nullptr;
  }

  try {
    auto* cpp_router = reinterpret_cast<Router*>(router);
    auto response = cpp_router->route(embedding, embedding_size, cost_bias);

    auto* result = static_cast<AdaptiveRouteResult*>(malloc(sizeof(AdaptiveRouteResult)));
    if (!result) {
      return nullptr;
    }

    result->selected_model = str_duplicate(response.selected_model);
    if (!result->selected_model) {
      free(result);
      return nullptr;
    }

    result->cluster_id = response.cluster_id;
    result->cluster_distance = response.cluster_distance;

    // Allocate alternatives array
    result->alternatives_count = response.alternatives.size();
    if (result->alternatives_count > 0) {
      result->alternatives
          = static_cast<char**>(malloc(sizeof(char*) * result->alternatives_count));
      if (!result->alternatives) {
        free(result->selected_model);
        free(result);
        return nullptr;
      }

      for (size_t alt_idx = 0; alt_idx < result->alternatives_count; ++alt_idx) {
        result->alternatives[alt_idx] = str_duplicate(response.alternatives[alt_idx]);
        if (!result->alternatives[alt_idx]) {
          // Cleanup previously allocated strings
          for (size_t cleanup_idx = 0; cleanup_idx < alt_idx; ++cleanup_idx) {
            free(result->alternatives[cleanup_idx]);
          }
          free(result->alternatives);
          free(result->selected_model);
          free(result);
          return nullptr;
        }
      }
    } else {
      result->alternatives = nullptr;
    }

    return result;
  } catch (const std::exception&) {
    return nullptr;
  }
}

char* adaptive_router_route_simple(AdaptiveRouter* router, const float* embedding,
                                   size_t embedding_size, float cost_bias) {
  if (!router || !embedding) {
    return nullptr;
  }

  try {
    auto* cpp_router = reinterpret_cast<Router*>(router);
    auto response = cpp_router->route(embedding, embedding_size, cost_bias);
    return str_duplicate(response.selected_model);
  } catch (const std::exception&) {
    return nullptr;
  }
}

void adaptive_route_result_free(AdaptiveRouteResult* result) {
  if (!result) return;
  cleanup_route_result_contents(result);
  free(result);
}

AdaptiveRouteResult* adaptive_router_route_double(AdaptiveRouter* router, const double* embedding,
                                                  size_t embedding_size, float cost_bias) {
  if (!router || !embedding) {
    return nullptr;
  }

  try {
    auto* cpp_router = reinterpret_cast<Router*>(router);
    // Use the templated route method
    auto response = cpp_router->route(embedding, embedding_size, cost_bias);

    auto* result = static_cast<AdaptiveRouteResult*>(malloc(sizeof(AdaptiveRouteResult)));
    if (!result) {
      return nullptr;
    }

    result->selected_model = str_duplicate(response.selected_model);
    if (!result->selected_model) {
      free(result);
      return nullptr;
    }

    result->cluster_id = response.cluster_id;
    result->cluster_distance = response.cluster_distance;

    // Allocate alternatives array
    result->alternatives_count = response.alternatives.size();
    if (result->alternatives_count > 0) {
      result->alternatives
          = static_cast<char**>(malloc(sizeof(char*) * result->alternatives_count));
      if (!result->alternatives) {
        free(result->selected_model);
        free(result);
        return nullptr;
      }

      for (size_t alt_idx = 0; alt_idx < result->alternatives_count; ++alt_idx) {
        result->alternatives[alt_idx] = str_duplicate(response.alternatives[alt_idx]);
        if (!result->alternatives[alt_idx]) {
          // Cleanup previously allocated strings
          for (size_t cleanup_idx = 0; cleanup_idx < alt_idx; ++cleanup_idx) {
            free(result->alternatives[cleanup_idx]);
          }
          free(result->alternatives);
          free(result->selected_model);
          free(result);
          return nullptr;
        }
      }
    } else {
      result->alternatives = nullptr;
    }

    return result;
  } catch (const std::exception&) {
    return nullptr;
  }
}

AdaptiveBatchRouteResult* adaptive_router_route_batch(
    AdaptiveRouter* router,
    const float* embeddings,
    size_t n_embeddings,
    size_t embedding_size,
    float cost_bias) {

  if (!router || !embeddings) {
    return nullptr;
  }

  try {
    auto* batch_result = static_cast<AdaptiveBatchRouteResult*>(malloc(sizeof(AdaptiveBatchRouteResult)));
    if (!batch_result) {
      return nullptr;
    }

    batch_result->count = n_embeddings;
    batch_result->results = nullptr;

    if (n_embeddings == 0) {
      return batch_result;
    }

    batch_result->results = static_cast<AdaptiveRouteResult*>(malloc(sizeof(AdaptiveRouteResult) * n_embeddings));
    if (!batch_result->results) {
      free(batch_result);
      return nullptr;
    }

    // Initialize all results to zero
    std::fill_n(batch_result->results, n_embeddings, AdaptiveRouteResult{});

    // Route each embedding
    for (size_t result_idx = 0; result_idx < n_embeddings; ++result_idx) {
      const float* embedding_ptr = embeddings + (result_idx * embedding_size);

      // Call single route
      auto* result = adaptive_router_route(router, embedding_ptr, embedding_size, cost_bias);

      if (result) {
        // Transfer ownership of data
        batch_result->results[result_idx] = *result;
        // Free only the result struct, not the data inside
        free(result);
      } else {
        // If routing fails, clean up previously allocated results
        for (size_t cleanup_idx = 0; cleanup_idx < result_idx; ++cleanup_idx) {
          cleanup_route_result_contents(&batch_result->results[cleanup_idx]);
        }
        free(batch_result->results);
        free(batch_result);
        return nullptr;
      }
    }

    return batch_result;
  } catch (const std::exception&) {
    return nullptr;
  }
}

void adaptive_batch_route_result_free(AdaptiveBatchRouteResult* result) {
  if (!result) return;

  if (result->results) {
    // Free each individual result's data using ranges
    auto results_span = std::span(result->results, result->count);
    std::ranges::for_each(results_span, [](AdaptiveRouteResult& res) {
      free(res.selected_model);
      if (res.alternatives) {
        auto alternatives_span = std::span(res.alternatives, res.alternatives_count);
        std::ranges::for_each(alternatives_span, [](char* str) { free(str); });
        free(res.alternatives);
      }
    });
    free(result->results);
  }
  free(result);
}

AdaptiveBatchRouteResult* adaptive_router_route_batch_double(
    AdaptiveRouter* router,
    const double* embeddings,
    size_t n_embeddings,
    size_t embedding_size,
    float cost_bias) {

  if (!router || !embeddings) {
    return nullptr;
  }

  try {
    auto* batch_result = static_cast<AdaptiveBatchRouteResult*>(malloc(sizeof(AdaptiveBatchRouteResult)));
    if (!batch_result) {
      return nullptr;
    }

    batch_result->count = n_embeddings;
    batch_result->results = nullptr;

    if (n_embeddings == 0) {
      return batch_result;
    }

    batch_result->results = static_cast<AdaptiveRouteResult*>(malloc(sizeof(AdaptiveRouteResult) * n_embeddings));
    if (!batch_result->results) {
      free(batch_result);
      return nullptr;
    }

    // Initialize all results to zero
    std::fill_n(batch_result->results, n_embeddings, AdaptiveRouteResult{});

    // Route each embedding using the double version
    for (size_t result_idx = 0; result_idx < n_embeddings; ++result_idx) {
      const double* embedding_ptr = embeddings + (result_idx * embedding_size);

      auto* result = adaptive_router_route_double(router, embedding_ptr, embedding_size, cost_bias);

      if (result) {
        batch_result->results[result_idx] = *result;
        free(result);
      } else {
        // If routing fails, clean up previously allocated results
        for (size_t cleanup_idx = 0; cleanup_idx < result_idx; ++cleanup_idx) {
          cleanup_route_result_contents(&batch_result->results[cleanup_idx]);
        }
        free(batch_result->results);
        free(batch_result);
        return nullptr;
      }
    }

    return batch_result;
  } catch (const std::exception&) {
    return nullptr;
  }
}

void adaptive_string_free(char* str) { free(str); }

size_t adaptive_router_get_n_clusters(AdaptiveRouter* router) {
  if (!router) {
    return 0;
  }

  try {
    auto* cpp_router = reinterpret_cast<Router*>(router);
    return static_cast<size_t>(cpp_router->get_n_clusters());
  } catch (const std::exception&) {
    return 0;
  }
}

size_t adaptive_router_get_embedding_dim(AdaptiveRouter* router) {
  if (!router) {
    return 0;
  }

  try {
    auto* cpp_router = reinterpret_cast<Router*>(router);
    return static_cast<size_t>(cpp_router->get_embedding_dim());
  } catch (const std::exception&) {
    return 0;
  }
}

char** adaptive_router_get_supported_models(AdaptiveRouter* router, size_t* count) {
  if (!router || !count) {
    if (count) *count = 0;
    return nullptr;
  }

  try {
    auto* cpp_router = reinterpret_cast<Router*>(router);
    auto models = cpp_router->get_supported_models();

    *count = models.size();
    if (models.empty()) {
      return nullptr;
    }

    char** result = static_cast<char**>(malloc(sizeof(char*) * models.size()));
    if (!result) {
      *count = 0;
      return nullptr;
    }

    for (size_t str_idx = 0; str_idx < models.size(); ++str_idx) {
      result[str_idx] = str_duplicate(models[str_idx]);
      if (!result[str_idx]) {
        // Allocation failed, clean up all previously allocated strings
        for (size_t cleanup_idx = 0; cleanup_idx < str_idx; ++cleanup_idx) {
          free(result[cleanup_idx]);
        }
        free(result);
        *count = 0;
        return nullptr;
      }
    }

    return result;
  } catch (const std::exception&) {
    *count = 0;
    return nullptr;
  }
}

void adaptive_string_array_free(char** strings, size_t count) {
  if (strings) {
    auto strings_span = std::span(strings, count);
    std::ranges::for_each(strings_span, [](char* str) { free(str); });
    free(strings);
  }
}

}  // extern "C"
