#ifndef RASMCORE_H
#define RASMCORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types */
typedef struct RasmPipeline RasmPipeline;
typedef struct RasmLayerCache RasmLayerCache;

/* Error handling (thread-local) */
const char* rasmcore_last_error(void);

/* Pipeline lifecycle */
RasmPipeline* rasmcore_pipeline_new(uint32_t cache_budget_mb);
void rasmcore_pipeline_free(RasmPipeline* pipe);

/* Layer cache (optional, cross-pipeline persistence) */
RasmLayerCache* rasmcore_layer_cache_new(uint32_t budget_mb);
void rasmcore_layer_cache_free(RasmLayerCache* cache);
void rasmcore_pipeline_set_cache(RasmPipeline* pipe, RasmLayerCache* cache);

/* Source */
uint32_t rasmcore_read(RasmPipeline* pipe, const uint8_t* data, size_t len);
uint32_t rasmcore_read_file(RasmPipeline* pipe, const char* path);

/* Filter dispatch (params as JSON) */
uint32_t rasmcore_filter(RasmPipeline* pipe, uint32_t source, const char* name, const char* params_json);

/* Output */
uint8_t* rasmcore_write(RasmPipeline* pipe, uint32_t node, const char* format, uint32_t quality, size_t* out_len);
int rasmcore_write_file(RasmPipeline* pipe, uint32_t node, const char* path, uint32_t quality);
void rasmcore_buffer_free(uint8_t* buf, size_t len);

/* Info */
const char* rasmcore_node_info_json(RasmPipeline* pipe, uint32_t node);

#ifdef __cplusplus
}
#endif

#endif /* RASMCORE_H */
