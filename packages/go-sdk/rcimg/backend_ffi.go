//go:build cgo

package rcimg

/*
#cgo LDFLAGS: -lrasmcore_ffi
#include <stdint.h>
#include <stdlib.h>

// Forward declarations matching rcimg.h
typedef struct rcimg_pipeline rcimg_pipeline;
extern rcimg_pipeline* rasmcore_pipeline_new(uint32_t cache_budget_mb);
extern void rasmcore_pipeline_free(rcimg_pipeline* pipe);
extern const char* rasmcore_last_error(void);
extern uint32_t rasmcore_read(rcimg_pipeline* pipe, const uint8_t* data, size_t len);
extern uint32_t rasmcore_filter(rcimg_pipeline* pipe, uint32_t source, const char* name, const char* params_json);
extern uint8_t* rasmcore_write(rcimg_pipeline* pipe, uint32_t node, const char* format, uint32_t quality, size_t* out_len);
extern void rasmcore_buffer_free(uint8_t* buf, size_t len);
*/
import "C"

import (
	"fmt"
	"strings"
	"unsafe"
)

type ffiBackend struct {
	pipe *C.rcimg_pipeline
}

func newFFIBackend(opts Options) (*ffiBackend, error) {
	budget := opts.CacheBudgetMB
	if budget == 0 {
		budget = 16
	}
	pipe := C.rasmcore_pipeline_new(C.uint32_t(budget))
	if pipe == nil {
		return nil, fmt.Errorf("FFI: rasmcore_pipeline_new failed: %s", C.GoString(C.rasmcore_last_error()))
	}
	return &ffiBackend{pipe: pipe}, nil
}

func (f *ffiBackend) Process(input []byte, ops []Op, output OutputConfig) ([]byte, error) {
	if len(ops) == 0 {
		return nil, fmt.Errorf("at least one operation is required")
	}
	if output.Format == "" {
		output.Format = "png"
	}

	// Read input image
	node := C.rasmcore_read(f.pipe, (*C.uint8_t)(unsafe.Pointer(&input[0])), C.size_t(len(input)))
	if node == C.UINT32_MAX {
		return nil, fmt.Errorf("read failed: %s", C.GoString(C.rasmcore_last_error()))
	}

	// Apply each operation
	for _, op := range ops {
		paramsJSON := "{}"
		if len(op.Params) > 0 {
			// Build JSON from positional params using manifest param names
			// For now, use the positional dispatch — the C API's rasmcore_filter
			// accepts JSON params by name, so we pass them as indexed keys
			parts := make([]string, len(op.Params))
			for i, v := range op.Params {
				parts[i] = fmt.Sprintf("\"p%d\": %g", i, v)
			}
			paramsJSON = "{" + strings.Join(parts, ", ") + "}"
		}

		cName := C.CString(op.Name)
		cParams := C.CString(paramsJSON)
		node = C.rasmcore_filter(f.pipe, node, cName, cParams)
		C.free(unsafe.Pointer(cName))
		C.free(unsafe.Pointer(cParams))

		if node == C.UINT32_MAX {
			return nil, fmt.Errorf("filter '%s' failed: %s", op.Name, C.GoString(C.rasmcore_last_error()))
		}
	}

	// Write output
	cFormat := C.CString(output.Format)
	var quality C.uint32_t
	if output.Quality != nil {
		quality = C.uint32_t(*output.Quality)
	}
	var outLen C.size_t
	outPtr := C.rasmcore_write(f.pipe, node, cFormat, quality, &outLen)
	C.free(unsafe.Pointer(cFormat))

	if outPtr == nil {
		return nil, fmt.Errorf("write failed: %s", C.GoString(C.rasmcore_last_error()))
	}

	// Copy to Go slice and free C buffer
	result := C.GoBytes(unsafe.Pointer(outPtr), C.int(outLen))
	C.rasmcore_buffer_free(outPtr, outLen)

	return result, nil
}

func (f *ffiBackend) Close() error {
	if f.pipe != nil {
		C.rasmcore_pipeline_free(f.pipe)
		f.pipe = nil
	}
	return nil
}
