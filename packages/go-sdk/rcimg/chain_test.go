package rcimg

import (
	"bytes"
	"os"
	"testing"
)

func TestFluentIdentity(t *testing.T) {
	wasmPath := os.Getenv("RASMCORE_WASM_PATH")
	if wasmPath == "" {
		t.Skip("RASMCORE_WASM_PATH not set; skipping integration test")
	}

	pipe, err := NewPipeline(Options{WASMPath: wasmPath})
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}

	input := makePNG()

	// Fluent API: invert x2 + flip x2 = identity
	result, err := pipe.Read(input).
		Invert().
		Invert().
		Flip(0.0).  // horizontal
		Flip(0.0).  // horizontal
		WritePNG()
	if err != nil {
		t.Fatalf("fluent chain: %v", err)
	}

	if !bytes.Equal(result[:4], []byte{0x89, 0x50, 0x4e, 0x47}) {
		t.Fatal("result is not a valid PNG")
	}
	t.Logf("fluent identity: %d bytes in → %d bytes out", len(input), len(result))
}

func TestFluentBlurSepia(t *testing.T) {
	wasmPath := os.Getenv("RASMCORE_WASM_PATH")
	if wasmPath == "" {
		t.Skip("RASMCORE_WASM_PATH not set; skipping integration test")
	}

	pipe, err := NewPipeline(Options{WASMPath: wasmPath})
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}

	input := makePNG()

	result, err := pipe.Read(input).
		Blur(2.0).
		Sepia(0.8).
		WriteJPEG(90)
	if err != nil {
		t.Fatalf("fluent chain: %v", err)
	}

	// JPEG starts with FF D8
	if len(result) < 2 || result[0] != 0xFF || result[1] != 0xD8 {
		t.Fatal("result is not a valid JPEG")
	}
	t.Logf("fluent blur+sepia: %d bytes in → %d bytes JPEG out", len(input), len(result))
}

func TestCompile(t *testing.T) {
	wasmPath := os.Getenv("RASMCORE_WASM_PATH")
	if wasmPath == "" {
		t.Skip("RASMCORE_WASM_PATH not set; skipping integration test")
	}

	pipe, err := NewPipeline(Options{WASMPath: wasmPath})
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}

	if err := pipe.Compile(); err != nil {
		t.Fatalf("Compile: %v", err)
	}

	// Verify .cwasm was created
	if pipe.cwasmPath == "" {
		t.Fatal("cwasmPath not set after Compile")
	}
	if _, err := os.Stat(pipe.cwasmPath); err != nil {
		t.Fatalf("cwasm file not found: %v", err)
	}

	// Run chain with precompiled module
	input := makePNG()
	result, err := pipe.Read(input).Invert().WritePNG()
	if err != nil {
		t.Fatalf("process with cwasm: %v", err)
	}
	if !bytes.Equal(result[:4], []byte{0x89, 0x50, 0x4e, 0x47}) {
		t.Fatal("result is not a valid PNG")
	}
	t.Logf("compiled pipeline: %d bytes out", len(result))

	// Clean up
	os.Remove(pipe.cwasmPath)
}
