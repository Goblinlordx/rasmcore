// Package rasmcore provides a Go SDK for the rasmcore WASM image processing pipeline.
//
// The SDK loads the rasmcore-image WASM component via wazero (pure Go, no CGO)
// and exposes a Pipeline type for image operations. Optional GPU acceleration
// is available via the gpu sub-package (requires wgpu-native CGO bindings).
//
// Usage:
//
//	pipe, err := rasmcore.NewPipeline(rasmcore.Options{})
//	if err != nil { log.Fatal(err) }
//	defer pipe.Close()
//
//	img, err := pipe.Read(imageBytes)
//	if err != nil { log.Fatal(err) }
//
//	blurred, err := pipe.Blur(img, rasmcore.BlurConfig{Radius: 10.0})
//	if err != nil { log.Fatal(err) }
//
//	result, err := pipe.WriteJPEG(blurred, rasmcore.JPEGConfig{Quality: 90})
//	if err != nil { log.Fatal(err) }
package rasmcore

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

// Options configures the Pipeline.
type Options struct {
	// WASMPath overrides the default WASM component location.
	// If empty, searches standard locations and RASMCORE_WASM_PATH env var.
	WASMPath string

	// UseGPU enables GPU acceleration if available. Default: true.
	UseGPU bool

	// GpuExecutor provides the GPU execution implementation.
	// If nil and UseGPU is true, attempts auto-detection.
	GpuExecutor GpuExecutor
}

// GpuExecutor is the interface for GPU compute offload.
// Implement this to provide GPU support (see gpu sub-package for wgpu-native impl).
type GpuExecutor interface {
	// Execute runs a batch of GPU compute operations on pixel data.
	// ops are chained: output of ops[i] = input of ops[i+1].
	Execute(ops []GpuOp, input []byte, width, height uint32) ([]byte, error)

	// MaxBufferSize returns the maximum buffer size in bytes.
	MaxBufferSize() int

	// Close releases GPU resources.
	Close()
}

// GpuOp represents a single GPU compute dispatch.
type GpuOp struct {
	Shader       string   // WGSL compute shader source
	EntryPoint   string   // Shader entry point name
	WorkgroupX   uint32   // Workgroup size X
	WorkgroupY   uint32   // Workgroup size Y
	WorkgroupZ   uint32   // Workgroup size Z
	Params       []byte   // Uniform parameters
	ExtraBuffers [][]byte // Additional storage buffers
}

// BlurConfig configures gaussian blur.
type BlurConfig struct {
	Radius float32
}

// SpherizeConfig configures spherize distortion.
type SpherizeConfig struct {
	Strength float32
}

// SepiaConfig configures sepia tone.
type SepiaConfig struct {
	Intensity float32
}

// JPEGConfig configures JPEG encoding.
type JPEGConfig struct {
	Quality uint8
	GPU     *bool // nil = use default, false = force CPU
}

// PNGConfig configures PNG encoding.
type PNGConfig struct {
	GPU *bool
}

// NodeID is a handle to a pipeline node.
type NodeID = uint32

// Pipeline wraps the rasmcore WASM image processing component.
type Pipeline struct {
	runtime wazero.Runtime
	module  api.Module
	ctx     context.Context
	gpu     GpuExecutor
}

func findWASM(override string) ([]byte, error) {
	candidates := []string{
		override,
		os.Getenv("RASMCORE_WASM_PATH"),
		filepath.Join(".", "rasmcore_image.wasm"),
		filepath.Join("target", "wasm32-wasip1", "release", "rasmcore_image.wasm"),
		filepath.Join("target", "wasm32-wasip1", "debug", "rasmcore_image.wasm"),
	}
	for _, p := range candidates {
		if p == "" {
			continue
		}
		data, err := os.ReadFile(p)
		if err == nil {
			return data, nil
		}
	}
	return nil, fmt.Errorf(
		"rasmcore_image.wasm not found; set RASMCORE_WASM_PATH or build with: " +
			"cargo component build -p rasmcore-image --release",
	)
}

// NewPipeline creates a new image processing pipeline.
//
// The WASM component is loaded once and reused for all operations.
// Call Close() when done to release resources.
func NewPipeline(opts Options) (*Pipeline, error) {
	ctx := context.Background()

	wasmBytes, err := findWASM(opts.WASMPath)
	if err != nil {
		return nil, err
	}

	rt := wazero.NewRuntime(ctx)

	// Instantiate WASI
	wasi_snapshot_preview1.MustInstantiate(ctx, rt)

	// TODO: Register gpu-execute host function if GPU available
	var gpu GpuExecutor
	if opts.UseGPU && opts.GpuExecutor != nil {
		gpu = opts.GpuExecutor
		// Register host function for gpu-execute WIT import
		// This requires component model support in wazero
	}

	// Compile and instantiate
	compiled, err := rt.CompileModule(ctx, wasmBytes)
	if err != nil {
		rt.Close(ctx)
		return nil, fmt.Errorf("compile WASM: %w", err)
	}

	mod, err := rt.InstantiateModule(ctx, compiled, wazero.NewModuleConfig())
	if err != nil {
		rt.Close(ctx)
		return nil, fmt.Errorf("instantiate WASM: %w", err)
	}

	return &Pipeline{
		runtime: rt,
		module:  mod,
		ctx:     ctx,
		gpu:     gpu,
	}, nil
}

// Close releases all pipeline resources.
func (p *Pipeline) Close() error {
	if p.gpu != nil {
		p.gpu.Close()
	}
	return p.runtime.Close(p.ctx)
}

// Read decodes image bytes (PNG, JPEG, etc.) into a pipeline node.
func (p *Pipeline) Read(imageBytes []byte) (NodeID, error) {
	// TODO: Call WASM pipeline.read() via component model
	// wazero doesn't support Component Model yet — this is a placeholder
	// for when wazero adds CM support or we use wasmtime-go instead
	return 0, fmt.Errorf("WASM Component Model not yet supported in wazero; " +
		"use wasmtime-go or wait for wazero CM support")
}

// Blur applies gaussian blur to a node.
func (p *Pipeline) Blur(source NodeID, config BlurConfig) (NodeID, error) {
	return 0, fmt.Errorf("not yet implemented — pending CM support")
}

// Spherize applies spherize distortion.
func (p *Pipeline) Spherize(source NodeID, config SpherizeConfig) (NodeID, error) {
	return 0, fmt.Errorf("not yet implemented — pending CM support")
}

// Sepia applies sepia tone.
func (p *Pipeline) Sepia(source NodeID, config SepiaConfig) (NodeID, error) {
	return 0, fmt.Errorf("not yet implemented — pending CM support")
}

// WriteJPEG encodes a node as JPEG.
func (p *Pipeline) WriteJPEG(source NodeID, config JPEGConfig) ([]byte, error) {
	return nil, fmt.Errorf("not yet implemented — pending CM support")
}

// WritePNG encodes a node as PNG.
func (p *Pipeline) WritePNG(source NodeID, config PNGConfig) ([]byte, error) {
	return nil, fmt.Errorf("not yet implemented — pending CM support")
}
