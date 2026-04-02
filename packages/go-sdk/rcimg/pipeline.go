// Package rcimg provides a Go SDK for the rasmcore image processing pipeline.
//
// Two backends are available:
//   - FFI (native): links librcimg via CGO. Native speed, GPU support, cross-call caching.
//     Build with CGO_ENABLED=1 and librcimg installed.
//   - WASM (universal): calls wasmtime CLI subprocess. Works everywhere, CPU only.
//     Requires wasmtime CLI and the rasmcore_image.wasm component.
//
// NewPipeline auto-selects FFI when available, falls back to WASM.
//
// Usage:
//
//	pipe, err := rcimg.NewPipeline(rcimg.Options{})
//	if err != nil { log.Fatal(err) }
//	defer pipe.Close()
//
//	result, err := pipe.Read(imageBytes).
//	    Blur(5.0).
//	    Sepia(0.8).
//	    WriteJPEG(90)
//	if err != nil { log.Fatal(err) }
//
//	os.WriteFile("output.jpg", result, 0644)
package rcimg

import "fmt"

// Options configures the Pipeline.
type Options struct {
	// WASMPath overrides the default WASM component path.
	WASMPath string

	// WasmtimePath overrides the wasmtime CLI binary path.
	WasmtimePath string

	// LibPath overrides the librcimg shared library path for FFI backend.
	// If empty, uses standard library search paths.
	LibPath string

	// ForceWASM disables FFI auto-detection and always uses the WASM backend.
	ForceWASM bool

	// CacheBudgetMB sets the spatial cache budget in megabytes (FFI only, default 16).
	CacheBudgetMB uint32
}

// Op represents a single operation in the processing chain.
type Op struct {
	Name   string    // Operation name (e.g., "invert", "blur", "flip")
	Params []float32 // Positional parameters (empty for no-config ops)
}

// OutputConfig controls the output encoding.
type OutputConfig struct {
	Format  string // Output format: "png", "jpeg", "webp", etc.
	Quality *uint8 // Optional quality (0-100) for lossy formats
}

// Pipeline wraps the rasmcore image processing engine.
// Auto-selects FFI (native) or WASM (subprocess) backend.
type Pipeline struct {
	backend Backend
	opts    Options
}

// NewPipeline creates a new image processing pipeline.
//
// Tries FFI backend first (if CGO enabled and librcimg available),
// falls back to WASM backend (wasmtime CLI).
func NewPipeline(opts Options) (*Pipeline, error) {
	if opts.CacheBudgetMB == 0 {
		opts.CacheBudgetMB = 16
	}

	// Try FFI backend unless forced to WASM
	if !opts.ForceWASM {
		if ffi, err := newFFIBackend(opts); err == nil {
			return &Pipeline{backend: ffi, opts: opts}, nil
		}
	}

	// Fall back to WASM
	wasm, err := newWasmtimeBackend(opts)
	if err != nil {
		return nil, fmt.Errorf("no backend available: %w", err)
	}
	return &Pipeline{backend: wasm, opts: opts}, nil
}

// Process runs a chain of operations on an image and returns the encoded result.
func (p *Pipeline) Process(input []byte, ops []Op, output OutputConfig) ([]byte, error) {
	return p.backend.Process(input, ops, output)
}

// Close releases all pipeline resources.
func (p *Pipeline) Close() error {
	return p.backend.Close()
}

// Compile pre-compiles the WASM for faster startup (WASM backend only).
// No-op for FFI backend.
func (p *Pipeline) Compile() error {
	if w, ok := p.backend.(*wasmtimeBackend); ok {
		return w.Compile()
	}
	return nil
}

// BackendName returns "ffi" or "wasm" indicating which backend is active.
func (p *Pipeline) BackendName() string {
	switch p.backend.(type) {
	case *ffiBackend:
		return "ffi"
	default:
		return "wasm"
	}
}
