// Package gpu provides GPU compute acceleration for rasmcore via wgpu-native.
//
// This package requires CGO and the wgpu-native shared library to be installed.
// If wgpu-native is not available, use rasmcore without GPU — the pipeline
// automatically falls back to CPU execution.
//
// Build tag: gpu
//
//go:build gpu
package gpu

/*
#cgo LDFLAGS: -lwgpu_native
// TODO: wgpu-native C header includes would go here
// #include <wgpu.h>
*/
import "C"

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/nicholasgasior/rasmcore-go/rasmcore"
)

// WgpuExecutor implements rasmcore.GpuExecutor using wgpu-native.
type WgpuExecutor struct {
	mu            sync.Mutex
	maxBufferSize int
	pipelineCache map[string]interface{} // shader hash → compiled pipeline
	// TODO: wgpu device, queue, adapter handles
}

// New creates a new GPU executor.
// Returns an error if no GPU adapter is found.
func New(opts ...Option) (*WgpuExecutor, error) {
	e := &WgpuExecutor{
		maxBufferSize: 256 * 1024 * 1024,
		pipelineCache: make(map[string]interface{}),
	}
	for _, opt := range opts {
		opt(e)
	}

	// TODO: Initialize wgpu-native
	// - Request adapter (high performance preference)
	// - Request device
	// - Store handles

	return e, nil
}

// Option configures the WgpuExecutor.
type Option func(*WgpuExecutor)

// WithMaxBufferSize sets the maximum GPU buffer size.
func WithMaxBufferSize(size int) Option {
	return func(e *WgpuExecutor) {
		e.maxBufferSize = size
	}
}

// Execute runs a batch of GPU compute operations.
func (e *WgpuExecutor) Execute(ops []rasmcore.GpuOp, input []byte, width, height uint32) ([]byte, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	byteLen := int(width) * int(height) * 4
	if byteLen > e.maxBufferSize {
		return nil, fmt.Errorf("image too large for GPU: %d bytes (max %d)", byteLen, e.maxBufferSize)
	}

	// TODO: Implement wgpu-native execution:
	// 1. Create/reuse storage buffers (ping-pong)
	// 2. Upload input via queue.writeBuffer
	// 3. For each op:
	//    a. Get or compile pipeline (cached by shader hash)
	//    b. Create bind group with buffers + params
	//    c. Dispatch compute
	//    d. Swap ping-pong buffers
	// 4. Read back output via buffer.mapAsync
	// 5. Return output bytes

	return nil, fmt.Errorf("wgpu-native execution not yet implemented")
}

// MaxBufferSize returns the maximum GPU buffer size.
func (e *WgpuExecutor) MaxBufferSize() int {
	return e.maxBufferSize
}

// Close releases GPU resources.
func (e *WgpuExecutor) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	// TODO: Release wgpu device, queue, adapter
	e.pipelineCache = nil
}

func shaderHash(source, entryPoint string) string {
	h := sha256.Sum256([]byte(source + ":" + entryPoint))
	return hex.EncodeToString(h[:8])
}
