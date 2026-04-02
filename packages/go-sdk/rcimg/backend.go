package rcimg

// Backend is the interface for pipeline execution backends.
// The WASM backend uses wasmtime CLI subprocess.
// The FFI backend uses CGO calls to librcimg (native speed + GPU).
type Backend interface {
	// Process runs a chain of operations and returns encoded output.
	Process(input []byte, ops []Op, output OutputConfig) ([]byte, error)

	// Close releases backend resources.
	Close() error
}
