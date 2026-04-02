// Package rcimg provides a Go SDK for the rasmcore WASM image processing pipeline.
//
// The SDK calls the rasmcore-image WASM component via the wasmtime CLI,
// using the process-chain function which accepts base64-encoded images
// and a list of operations. This works with any language that can exec
// a subprocess — no Component Model host bindings required.
//
// Usage:
//
//	pipe, err := rcimg.NewPipeline(rcimg.Options{})
//	if err != nil { log.Fatal(err) }
//
//	result, err := pipe.Process(imageBytes, []rcimg.Op{
//	    {Name: "invert"},
//	    {Name: "blur", Params: []float32{5.0}},
//	    {Name: "flip", Params: []float32{0.0}},
//	}, rcimg.OutputConfig{Format: "png"})
//	if err != nil { log.Fatal(err) }
//
//	os.WriteFile("output.png", result, 0644)
package rcimg

import (
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Options configures the Pipeline.
type Options struct {
	// WASMPath overrides the default WASM component path.
	// If empty, searches standard locations and RASMCORE_WASM_PATH env var.
	WASMPath string

	// WasmtimePath overrides the wasmtime CLI binary path.
	// If empty, uses "wasmtime" from PATH.
	WasmtimePath string
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

// Pipeline wraps the rasmcore WASM component via the wasmtime CLI.
type Pipeline struct {
	wasmPath     string
	cwasmPath    string // precompiled .cwasm path (set by Compile)
	wasmtimePath string
}

func findWASM(override string) (string, error) {
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
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf(
		"rasmcore_image.wasm not found; set RASMCORE_WASM_PATH or build with: " +
			"cargo component build -p rasmcore-image --release",
	)
}

func findWasmtime(override string) (string, error) {
	if override != "" {
		if _, err := os.Stat(override); err == nil {
			return override, nil
		}
		return "", fmt.Errorf("wasmtime not found at %q", override)
	}
	path, err := exec.LookPath("wasmtime")
	if err != nil {
		return "", fmt.Errorf("wasmtime CLI not found in PATH; install from https://wasmtime.dev")
	}
	return path, nil
}

// NewPipeline creates a new image processing pipeline.
//
// Requires wasmtime CLI to be installed and the rasmcore_image.wasm component
// to be built. Each Process call spawns a wasmtime subprocess.
func NewPipeline(opts Options) (*Pipeline, error) {
	wasmPath, err := findWASM(opts.WASMPath)
	if err != nil {
		return nil, err
	}
	wtPath, err := findWasmtime(opts.WasmtimePath)
	if err != nil {
		return nil, err
	}
	return &Pipeline{
		wasmPath:     wasmPath,
		wasmtimePath: wtPath,
	}, nil
}

// Process runs a chain of operations on an image and returns the encoded result.
//
// Input is raw image bytes (PNG, JPEG, etc.). Operations are applied in order.
// Output is encoded in the specified format.
func (p *Pipeline) Process(input []byte, ops []Op, output OutputConfig) ([]byte, error) {
	if len(ops) == 0 {
		return nil, fmt.Errorf("at least one operation is required")
	}
	if output.Format == "" {
		output.Format = "png"
	}

	// Base64 encode input
	inputB64 := base64.StdEncoding.EncodeToString(input)

	// Build WAVE ops list
	opStrs := make([]string, len(ops))
	for i, op := range ops {
		paramStrs := make([]string, len(op.Params))
		for j, v := range op.Params {
			paramStrs[j] = fmt.Sprintf("%g", v)
		}
		opStrs[i] = fmt.Sprintf("{name: \"%s\", params: [%s]}", op.Name, strings.Join(paramStrs, ", "))
	}

	// Quality in WAVE format
	qualityWave := "none"
	if output.Quality != nil {
		qualityWave = fmt.Sprintf("some(%d)", *output.Quality)
	}

	// Build the WAVE invocation string
	invoke := fmt.Sprintf(
		`process-chain("%s", [%s], "%s", %s)`,
		inputB64,
		strings.Join(opStrs, ", "),
		output.Format,
		qualityWave,
	)

	// Execute wasmtime (use precompiled .cwasm if available)
	runPath := p.wasmPath
	args := []string{"run", "--invoke", invoke}
	if p.cwasmPath != "" {
		runPath = p.cwasmPath
		args = append(args, "--allow-precompiled")
	}
	args = append(args, runPath)
	cmd := exec.Command(p.wasmtimePath, args...)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("wasmtime failed: %s\n%s", ee.Error(), string(ee.Stderr))
		}
		return nil, fmt.Errorf("wasmtime failed: %w", err)
	}

	// Parse WAVE result: ok("...base64...") or err(variant("message"))
	result := strings.TrimSpace(string(out))
	if strings.HasPrefix(result, `ok("`) && strings.HasSuffix(result, `")`) {
		b64 := result[4 : len(result)-2]
		return base64.StdEncoding.DecodeString(b64)
	}
	if strings.HasPrefix(result, "err(") {
		return nil, fmt.Errorf("pipeline error: %s", result[4:len(result)-1])
	}
	return nil, fmt.Errorf("unexpected wasmtime output: %s", result)
}

// Compile pre-compiles the WASM component to native code via `wasmtime compile`.
// This eliminates cold-start latency on subsequent Process() calls.
// The compiled .cwasm file is stored alongside the .wasm file.
func (p *Pipeline) Compile() error {
	cwasmPath := p.wasmPath + ".cwasm"
	// Check if already compiled and newer than source
	if info, err := os.Stat(cwasmPath); err == nil {
		if srcInfo, err2 := os.Stat(p.wasmPath); err2 == nil {
			if info.ModTime().After(srcInfo.ModTime()) {
				p.cwasmPath = cwasmPath
				return nil
			}
		}
	}
	cmd := exec.Command(p.wasmtimePath, "compile", p.wasmPath, "-o", cwasmPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("wasmtime compile failed: %s\n%s", err, string(out))
	}
	p.cwasmPath = cwasmPath
	return nil
}

// Close is a no-op for the CLI-based pipeline (no persistent resources).
func (p *Pipeline) Close() error {
	return nil
}
