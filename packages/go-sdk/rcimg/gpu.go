package rcimg

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
