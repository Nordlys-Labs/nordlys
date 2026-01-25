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
 * Route result structure
 */
typedef struct {
  char* selected_model;      /**< Selected model ID (caller must free with nordlys_string_free) */
  char** alternatives;       /**< Array of alternative model IDs */
  size_t alternatives_count; /**< Number of alternatives */
  int cluster_id;            /**< Assigned cluster ID */
  float cluster_distance;    /**< Distance to cluster centroid */
} NordlysRouteResult;

/**
 * Batch route result structure
 */
typedef struct {
  NordlysRouteResult* results; /**< Array of route results */
  size_t count;                /**< Number of results */
} NordlysBatchRouteResult;

/**
 * Device type for cluster backend
 */
typedef enum { NORDLYS_DEVICE_CPU = 0, NORDLYS_DEVICE_CUDA = 1 } NordlysDevice;

/**
 * Error codes for nordlys router operations
 */
typedef enum {
  NORDLYS_OK = 0,
  NORDLYS_ERROR_NULL_ROUTER,
  NORDLYS_ERROR_NULL_EMBEDDING,
  NORDLYS_ERROR_DIMENSION_MISMATCH,
  NORDLYS_ERROR_ALLOCATION_FAILED,
  NORDLYS_ERROR_INTERNAL
} NordlysErrorCode;

/**
 * Create a router from a JSON profile file
 * @param profile_path Path to the JSON profile file
 * @param device Device to use (NORDLYS_DEVICE_CPU or NORDLYS_DEVICE_CUDA)
 * @return Router handle, or NULL on error
 */
NORDLYS_API NordlysRouter* nordlys_router_create(const char* profile_path, NordlysDevice device);

/**
 * Create a router from a JSON string
 * @param json_str JSON string containing the profile
 * @param device Device to use (NORDLYS_DEVICE_CPU or NORDLYS_DEVICE_CUDA)
 * @return Router handle, or NULL on error
 */
NORDLYS_API NordlysRouter* nordlys_router_create_from_json(const char* json_str,
                                                           NordlysDevice device);

/**
 * Create a router from a binary MessagePack file
 * @param path Path to the binary profile file
 * @param device Device to use (NORDLYS_DEVICE_CPU or NORDLYS_DEVICE_CUDA)
 * @return Router handle, or NULL on error
 */
NORDLYS_API NordlysRouter* nordlys_router_create_from_msgpack(const char* path,
                                                              NordlysDevice device);

/**
 * Destroy a router and free its resources
 * @param router Router handle
 */
NORDLYS_API void nordlys_router_destroy(NordlysRouter* router);

/**
 * Route using a pre-computed embedding
 * @param router Router handle
 * @param embedding Pointer to embedding data (float array)
 * @param embedding_size Size of the embedding array
 * @param error_out Optional error code output (can be NULL)
 * @return Route result (caller must free with nordlys_route_result_free)
 */
NORDLYS_API NordlysRouteResult* nordlys_router_route(NordlysRouter* router, const float* embedding,
                                                     size_t embedding_size,
                                                     NordlysErrorCode* error_out);

/**
 * Batch route using multiple pre-computed embeddings
 * @param router Router handle
 * @param embeddings Pointer to embedding data (N×D row-major array)
 * @param n_embeddings Number of embeddings in batch
 * @param embedding_size Dimension of each embedding (D)
 * @param error_out Optional error code output (can be NULL)
 * @return Batch route result (caller must free with nordlys_batch_route_result_free)
 */
NORDLYS_API NordlysBatchRouteResult* nordlys_router_route_batch(NordlysRouter* router,
                                                                const float* embeddings,
                                                                size_t n_embeddings,
                                                                size_t embedding_size,
                                                                NordlysErrorCode* error_out);

/**
 * Free a route result
 * @param result Route result to free
 */
NORDLYS_API void nordlys_route_result_free(NordlysRouteResult* result);

/**
 * Free a batch route result
 * @param result Batch result to free
 */
NORDLYS_API void nordlys_batch_route_result_free(NordlysBatchRouteResult* result);

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
 * Free an array of strings
 * @param strings Array to free
 * @param count Number of strings in array
 */
NORDLYS_API void nordlys_string_array_free(char** strings, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* NORDLYS_H */
