#ifndef NORDLYS_H
#define NORDLYS_H

#include <stddef.h>

/* Cross-platform DLL export/import macros */
#if defined(_WIN32) || defined(_WIN64)
#  ifdef NORDLYS_C_EXPORTS
#    define NORDLYS_API __declspec(dllexport)
#  else
#    define NORDLYS_API __declspec(dllimport)
#  endif
#else
#  if __GNUC__ >= 4
#    define NORDLYS_API __attribute__((visibility("default")))
#  else
#    define NORDLYS_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to a Router instance
 */
typedef struct NordlysRouter NordlysRouter;

/**
  * Route result structure (float32)
  */
typedef struct {
    char* selected_model;      /**< Selected model ID (caller must free with nordlys_string_free) */
    char** alternatives;       /**< Array of alternative model IDs */
    size_t alternatives_count; /**< Number of alternatives */
    int cluster_id;            /**< Assigned cluster ID */
    float cluster_distance;    /**< Distance to cluster centroid */
} NordlysRouteResult32;

/**
  * Route result structure (float64)
  */
typedef struct {
    char* selected_model;      /**< Selected model ID (caller must free with nordlys_string_free) */
    char** alternatives;       /**< Array of alternative model IDs */
    size_t alternatives_count; /**< Number of alternatives */
    int cluster_id;            /**< Assigned cluster ID */
    double cluster_distance;   /**< Distance to cluster centroid */
} NordlysRouteResult64;

/**
  * Batch route result structure (float32)
  */
typedef struct {
   NordlysRouteResult32* results;  /**< Array of route results */
   size_t count;                     /**< Number of results */
} NordlysBatchRouteResult32;

/**
  * Batch route result structure (float64)
  */
typedef struct {
   NordlysRouteResult64* results;  /**< Array of route results */
   size_t count;                     /**< Number of results */
} NordlysBatchRouteResult64;

/**
 * Router precision type
 */
typedef enum {
  NORDLYS_PRECISION_FLOAT32 = 0,
  NORDLYS_PRECISION_FLOAT64 = 1,
  NORDLYS_PRECISION_UNKNOWN = -1
} NordlysPrecision;

/**
 * Error codes for nordlys router operations
 */
typedef enum {
  NORDLYS_OK = 0,
  NORDLYS_ERROR_NULL_ROUTER,
  NORDLYS_ERROR_NULL_EMBEDDING,
  NORDLYS_ERROR_TYPE_MISMATCH,
  NORDLYS_ERROR_DIMENSION_MISMATCH,
  NORDLYS_ERROR_ALLOCATION_FAILED,
  NORDLYS_ERROR_INTERNAL
} NordlysErrorCode;

/**
 * Create a router from a JSON profile file
 * @param profile_path Path to the JSON profile file
 * @return Router handle, or NULL on error
 */
NORDLYS_API NordlysRouter* nordlys_router_create(const char* profile_path);

/**
 * Create a router from a JSON string
 * @param json_str JSON string containing the profile
 * @return Router handle, or NULL on error
 */
NORDLYS_API NordlysRouter* nordlys_router_create_from_json(const char* json_str);

/**
 * Create a router from a binary MessagePack file
 * @param path Path to the binary profile file
 * @return Router handle, or NULL on error
 */
NORDLYS_API NordlysRouter* nordlys_router_create_from_binary(const char* path);

/**
 * Destroy a router and free its resources
 * @param router Router handle
 */
NORDLYS_API void nordlys_router_destroy(NordlysRouter* router);

/**
  * Route using a pre-computed embedding (float32)
  * @param router Router handle
  * @param embedding Pointer to embedding data (float array)
  * @param embedding_size Size of the embedding array
  * @param cost_bias Cost bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  * @param error_out Optional error code output (can be NULL)
  * @return Route result (caller must free with nordlys_route_result_free_f32)
  */
NORDLYS_API NordlysRouteResult32* nordlys_router_route_f32(NordlysRouter* router,
                                                              const float* embedding,
                                                              size_t embedding_size,
                                                              float cost_bias,
                                                              NordlysErrorCode* error_out);

/**
  * Route using a pre-computed embedding (float64)
  * @param router Router handle
  * @param embedding Pointer to embedding data (double array)
  * @param embedding_size Size of the embedding array
  * @param cost_bias Cost bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  * @param error_out Optional error code output (can be NULL)
  * @return Route result (caller must free with nordlys_route_result_free_f64)
  */
NORDLYS_API NordlysRouteResult64* nordlys_router_route_f64(NordlysRouter* router,
                                                              const double* embedding,
                                                              size_t embedding_size,
                                                              float cost_bias,
                                                              NordlysErrorCode* error_out);

/**
  * Batch route using multiple pre-computed embeddings (float32)
  * @param router Router handle
  * @param embeddings Pointer to embedding data (N×D row-major array)
  * @param n_embeddings Number of embeddings in batch
  * @param embedding_size Dimension of each embedding (D)
  * @param cost_bias Cost bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  * @param error_out Optional error code output (can be NULL)
  * @return Batch route result (caller must free with nordlys_batch_route_result_free_f32)
  */
NORDLYS_API NordlysBatchRouteResult32* nordlys_router_route_batch_f32(
    NordlysRouter* router,
    const float* embeddings,
    size_t n_embeddings,
    size_t embedding_size,
    float cost_bias,
    NordlysErrorCode* error_out);

/**
  * Batch route using multiple pre-computed embeddings (float64)
  * @param router Router handle
  * @param embeddings Pointer to embedding data (N×D row-major array)
  * @param n_embeddings Number of embeddings in batch
  * @param embedding_size Dimension of each embedding (D)
  * @param cost_bias Cost bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  * @param error_out Optional error code output (can be NULL)
  * @return Batch route result (caller must free with nordlys_batch_route_result_free_f64)
  */
NORDLYS_API NordlysBatchRouteResult64* nordlys_router_route_batch_f64(
    NordlysRouter* router,
    const double* embeddings,
    size_t n_embeddings,
    size_t embedding_size,
    float cost_bias,
    NordlysErrorCode* error_out);

/**
  * Free a route result (float32)
  * @param result Route result to free
  */
NORDLYS_API void nordlys_route_result_free_f32(NordlysRouteResult32* result);

/**
  * Free a route result (float64)
  * @param result Route result to free
  */
NORDLYS_API void nordlys_route_result_free_f64(NordlysRouteResult64* result);

/**
  * Free a batch route result (float32)
  * @param result Batch result to free
  */
NORDLYS_API void nordlys_batch_route_result_free_f32(NordlysBatchRouteResult32* result);

/**
  * Free a batch route result (float64)
  * @param result Batch result to free
  */
NORDLYS_API void nordlys_batch_route_result_free_f64(NordlysBatchRouteResult64* result);

/**
 * Free a string returned by the API
 * @param str String to free
 */
NORDLYS_API void nordlys_string_free(char* str);

/**
 * Get number of clusters
 * @param router Router handle
 * @return Number of clusters
 */
NORDLYS_API size_t nordlys_router_get_n_clusters(NordlysRouter* router);

/**
 * Get expected embedding dimension
 * @param router Router handle
 * @return Embedding dimension
 */
NORDLYS_API size_t nordlys_router_get_embedding_dim(NordlysRouter* router);

/**
 * Get supported models
 * @param router Router handle
 * @param count Output parameter for number of models
 * @return Array of model IDs (caller must free each string and the array)
 */
NORDLYS_API char** nordlys_router_get_supported_models(NordlysRouter* router, size_t* count);

/**
 * Get router precision type
 * @param router Router handle
 * @return ADAPTIVE_PRECISION_FLOAT32, ADAPTIVE_PRECISION_FLOAT64, or ADAPTIVE_PRECISION_UNKNOWN if router is NULL
 */
NORDLYS_API NordlysPrecision nordlys_router_get_precision(NordlysRouter* router);

/**
 * Free an array of strings
 * @param strings Array to free
 * @param count Number of strings in array
 */
NORDLYS_API void nordlys_string_array_free(char** strings, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* NORDLYS_H */
