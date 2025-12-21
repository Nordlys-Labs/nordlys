#ifndef ADAPTIVE_ROUTER_H
#define ADAPTIVE_ROUTER_H

#include <stddef.h>

/* Cross-platform DLL export/import macros */
#if defined(_WIN32) || defined(_WIN64)
#  ifdef ADAPTIVE_C_EXPORTS
#    define ADAPTIVE_API __declspec(dllexport)
#  else
#    define ADAPTIVE_API __declspec(dllimport)
#  endif
#else
#  if __GNUC__ >= 4
#    define ADAPTIVE_API __attribute__((visibility("default")))
#  else
#    define ADAPTIVE_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to a Router instance
 */
typedef struct AdaptiveRouter AdaptiveRouter;

/**
  * Route result structure (float32)
  */
typedef struct {
    char* selected_model;      /**< Selected model ID (caller must free with adaptive_string_free) */
    char** alternatives;       /**< Array of alternative model IDs */
    size_t alternatives_count; /**< Number of alternatives */
    int cluster_id;            /**< Assigned cluster ID */
    float cluster_distance;    /**< Distance to cluster centroid */
} AdaptiveRouteResult32;

/**
  * Route result structure (float64)
  */
typedef struct {
    char* selected_model;      /**< Selected model ID (caller must free with adaptive_string_free) */
    char** alternatives;       /**< Array of alternative model IDs */
    size_t alternatives_count; /**< Number of alternatives */
    int cluster_id;            /**< Assigned cluster ID */
    double cluster_distance;   /**< Distance to cluster centroid */
} AdaptiveRouteResult64;

/**
  * Batch route result structure (float32)
  */
typedef struct {
   AdaptiveRouteResult32* results;  /**< Array of route results */
   size_t count;                     /**< Number of results */
} AdaptiveBatchRouteResult32;

/**
  * Batch route result structure (float64)
  */
typedef struct {
   AdaptiveRouteResult64* results;  /**< Array of route results */
   size_t count;                     /**< Number of results */
} AdaptiveBatchRouteResult64;

/**
 * Router precision type
 */
typedef enum {
  ADAPTIVE_PRECISION_FLOAT32 = 0,
  ADAPTIVE_PRECISION_FLOAT64 = 1,
  ADAPTIVE_PRECISION_UNKNOWN = -1
} AdaptivePrecision;

/**
 * Error codes for adaptive router operations
 */
typedef enum {
  ADAPTIVE_OK = 0,
  ADAPTIVE_ERROR_NULL_ROUTER,
  ADAPTIVE_ERROR_NULL_EMBEDDING,
  ADAPTIVE_ERROR_TYPE_MISMATCH,
  ADAPTIVE_ERROR_DIMENSION_MISMATCH,
  ADAPTIVE_ERROR_ALLOCATION_FAILED,
  ADAPTIVE_ERROR_INTERNAL
} AdaptiveErrorCode;

/**
 * Create a router from a JSON profile file
 * @param profile_path Path to the JSON profile file
 * @return Router handle, or NULL on error
 */
ADAPTIVE_API AdaptiveRouter* adaptive_router_create(const char* profile_path);

/**
 * Create a router from a JSON string
 * @param json_str JSON string containing the profile
 * @return Router handle, or NULL on error
 */
ADAPTIVE_API AdaptiveRouter* adaptive_router_create_from_json(const char* json_str);

/**
 * Create a router from a binary MessagePack file
 * @param path Path to the binary profile file
 * @return Router handle, or NULL on error
 */
ADAPTIVE_API AdaptiveRouter* adaptive_router_create_from_binary(const char* path);

/**
 * Destroy a router and free its resources
 * @param router Router handle
 */
ADAPTIVE_API void adaptive_router_destroy(AdaptiveRouter* router);

/**
  * Route using a pre-computed embedding (float32)
  * @param router Router handle
  * @param embedding Pointer to embedding data (float array)
  * @param embedding_size Size of the embedding array
  * @param cost_bias Cost bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  * @param error_out Optional error code output (can be NULL)
  * @return Route result (caller must free with adaptive_route_result_free_f32)
  */
ADAPTIVE_API AdaptiveRouteResult32* adaptive_router_route_f32(AdaptiveRouter* router,
                                                              const float* embedding,
                                                              size_t embedding_size,
                                                              float cost_bias,
                                                              AdaptiveErrorCode* error_out);

/**
  * Route using a pre-computed embedding (float64)
  * @param router Router handle
  * @param embedding Pointer to embedding data (double array)
  * @param embedding_size Size of the embedding array
  * @param cost_bias Cost bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  * @param error_out Optional error code output (can be NULL)
  * @return Route result (caller must free with adaptive_route_result_free_f64)
  */
ADAPTIVE_API AdaptiveRouteResult64* adaptive_router_route_f64(AdaptiveRouter* router,
                                                              const double* embedding,
                                                              size_t embedding_size,
                                                              float cost_bias,
                                                              AdaptiveErrorCode* error_out);

/**
  * Batch route using multiple pre-computed embeddings (float32)
  * @param router Router handle
  * @param embeddings Pointer to embedding data (N×D row-major array)
  * @param n_embeddings Number of embeddings in batch
  * @param embedding_size Dimension of each embedding (D)
  * @param cost_bias Cost bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  * @param error_out Optional error code output (can be NULL)
  * @return Batch route result (caller must free with adaptive_batch_route_result_free_f32)
  */
ADAPTIVE_API AdaptiveBatchRouteResult32* adaptive_router_route_batch_f32(
    AdaptiveRouter* router,
    const float* embeddings,
    size_t n_embeddings,
    size_t embedding_size,
    float cost_bias,
    AdaptiveErrorCode* error_out);

/**
  * Batch route using multiple pre-computed embeddings (float64)
  * @param router Router handle
  * @param embeddings Pointer to embedding data (N×D row-major array)
  * @param n_embeddings Number of embeddings in batch
  * @param embedding_size Dimension of each embedding (D)
  * @param cost_bias Cost bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  * @param error_out Optional error code output (can be NULL)
  * @return Batch route result (caller must free with adaptive_batch_route_result_free_f64)
  */
ADAPTIVE_API AdaptiveBatchRouteResult64* adaptive_router_route_batch_f64(
    AdaptiveRouter* router,
    const double* embeddings,
    size_t n_embeddings,
    size_t embedding_size,
    float cost_bias,
    AdaptiveErrorCode* error_out);

/**
  * Free a route result (float32)
  * @param result Route result to free
  */
ADAPTIVE_API void adaptive_route_result_free_f32(AdaptiveRouteResult32* result);

/**
  * Free a route result (float64)
  * @param result Route result to free
  */
ADAPTIVE_API void adaptive_route_result_free_f64(AdaptiveRouteResult64* result);

/**
  * Free a batch route result (float32)
  * @param result Batch result to free
  */
ADAPTIVE_API void adaptive_batch_route_result_free_f32(AdaptiveBatchRouteResult32* result);

/**
  * Free a batch route result (float64)
  * @param result Batch result to free
  */
ADAPTIVE_API void adaptive_batch_route_result_free_f64(AdaptiveBatchRouteResult64* result);

/**
 * Free a string returned by the API
 * @param str String to free
 */
ADAPTIVE_API void adaptive_string_free(char* str);

/**
 * Get number of clusters
 * @param router Router handle
 * @return Number of clusters
 */
ADAPTIVE_API size_t adaptive_router_get_n_clusters(AdaptiveRouter* router);

/**
 * Get expected embedding dimension
 * @param router Router handle
 * @return Embedding dimension
 */
ADAPTIVE_API size_t adaptive_router_get_embedding_dim(AdaptiveRouter* router);

/**
 * Get supported models
 * @param router Router handle
 * @param count Output parameter for number of models
 * @return Array of model IDs (caller must free each string and the array)
 */
ADAPTIVE_API char** adaptive_router_get_supported_models(AdaptiveRouter* router, size_t* count);

/**
 * Get router precision type
 * @param router Router handle
 * @return ADAPTIVE_PRECISION_FLOAT32, ADAPTIVE_PRECISION_FLOAT64, or ADAPTIVE_PRECISION_UNKNOWN if router is NULL
 */
ADAPTIVE_API AdaptivePrecision adaptive_router_get_precision(AdaptiveRouter* router);

/**
 * Free an array of strings
 * @param strings Array to free
 * @param count Number of strings in array
 */
ADAPTIVE_API void adaptive_string_array_free(char** strings, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* ADAPTIVE_ROUTER_H */
