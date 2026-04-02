/* Minimal smoke test for librcimg C API. */
#include <stdio.h>
#include <assert.h>
#include "../include/rcimg.h"

int main(void) {
    /* Create and free a pipeline — verifies linking works. */
    rcimg_pipeline *pipe = rasmcore_pipeline_new(4);
    assert(pipe != NULL);
    printf("pipeline created: %p\n", (void *)pipe);

    /* No image loaded yet — reading info should fail gracefully. */
    const char *info = rasmcore_node_info_json(pipe, 0);
    if (info == NULL) {
        printf("node_info(0) = NULL (expected, no nodes yet)\n");
        printf("last_error: %s\n", rasmcore_last_error());
    }

    rasmcore_pipeline_free(pipe);
    printf("pipeline freed\n");

    /* NULL-safety — free(NULL) should not crash. */
    rasmcore_pipeline_free(NULL);
    rasmcore_buffer_free(NULL, 0);
    printf("NULL-safety OK\n");

    printf("PASS\n");
    return 0;
}
