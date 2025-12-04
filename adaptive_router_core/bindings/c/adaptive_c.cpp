#include "adaptive.h"
#include "router.hpp"
#include <cstring>
#include <exception>
#include <memory>

// Internal helper to convert std::string to C string
static char* str_duplicate(const std::string& str) {
    char* result = static_cast<char*>(malloc(str.length() + 1));
    if (result) {
        std::strcpy(result, str.c_str());
    }
    return result;
}

// C API implementation
extern "C" {

AdaptiveRouter* adaptive_router_create(const char* profile_path) {
    try {
        auto* router = new Router(
            Router::from_file(profile_path)
        );
        return reinterpret_cast<AdaptiveRouter*>(router);
    } catch (const std::exception&) {
        return nullptr;
    }
}

AdaptiveRouter* adaptive_router_create_from_json(const char* json_str) {
    try {
        auto* router = new Router(
            Router::from_json_string(json_str)
        );
        return reinterpret_cast<AdaptiveRouter*>(router);
    } catch (const std::exception&) {
        return nullptr;
    }
}

AdaptiveRouter* adaptive_router_create_from_binary(const char* path) {
    try {
        auto* router = new Router(
            Router::from_binary(path)
        );
        return reinterpret_cast<AdaptiveRouter*>(router);
    } catch (const std::exception&) {
        return nullptr;
    }
}

void adaptive_router_destroy(AdaptiveRouter* router) {
    if (router) {
        delete reinterpret_cast<Router*>(router);
    }
}

AdaptiveRouteResult* adaptive_router_route(
    AdaptiveRouter* router,
    const float* embedding,
    size_t embedding_size,
    float cost_bias
) {
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
        result->cluster_id = response.cluster_id;
        result->cluster_distance = response.cluster_distance;

        // Allocate alternatives array
        result->alternatives_count = response.alternatives.size();
        if (result->alternatives_count > 0) {
            result->alternatives = static_cast<char**>(
                malloc(sizeof(char*) * result->alternatives_count)
            );
            if (result->alternatives) {
                for (size_t i = 0; i < result->alternatives_count; ++i) {
                    result->alternatives[i] = str_duplicate(response.alternatives[i]);
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

char* adaptive_router_route_simple(
    AdaptiveRouter* router,
    const float* embedding,
    size_t embedding_size,
    float cost_bias
) {
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
    if (result) {
        free(result->selected_model);
        if (result->alternatives) {
            for (size_t i = 0; i < result->alternatives_count; ++i) {
                free(result->alternatives[i]);
            }
            free(result->alternatives);
        }
        free(result);
    }
}

void adaptive_string_free(char* str) {
    free(str);
}

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

        for (size_t i = 0; i < models.size(); ++i) {
            result[i] = str_duplicate(models[i]);
        }

        return result;
    } catch (const std::exception&) {
        *count = 0;
        return nullptr;
    }
}

void adaptive_string_array_free(char** strings, size_t count) {
    if (strings) {
        for (size_t i = 0; i < count; ++i) {
            free(strings[i]);
        }
        free(strings);
    }
}

}  // extern "C"
