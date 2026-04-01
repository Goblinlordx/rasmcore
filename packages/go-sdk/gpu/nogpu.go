// Package gpu — stub when built without GPU support.
//
//go:build !gpu
package gpu

import (
	"fmt"

	"github.com/nicholasgasior/rcimg-go/rcimg"
)

// WgpuExecutor stub — returns error when GPU build tag not set.
type WgpuExecutor struct{}

// New returns an error — GPU support requires the "gpu" build tag
// and wgpu-native library.
func New(opts ...Option) (*WgpuExecutor, error) {
	return nil, fmt.Errorf("GPU support not available: rebuild with -tags gpu and wgpu-native installed")
}

// Option is a no-op in the stub.
type Option func(*WgpuExecutor)

// WithMaxBufferSize is a no-op in the stub.
func WithMaxBufferSize(size int) Option {
	return func(*WgpuExecutor) {}
}

// Execute always returns an error.
func (e *WgpuExecutor) Execute(ops []rcimg.GpuOp, input []byte, width, height uint32) ([]byte, error) {
	return nil, fmt.Errorf("GPU not available")
}

// MaxBufferSize returns 0.
func (e *WgpuExecutor) MaxBufferSize() int { return 0 }

// Close is a no-op.
func (e *WgpuExecutor) Close() {}
