package rcimg

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"hash/crc32"
	"os"
	"testing"
)

// makePNG creates a minimal 4x4 red RGB PNG for testing.
func makePNG() []byte {
	var buf bytes.Buffer

	// PNG signature
	buf.Write([]byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a})

	writeChunk := func(ctype string, data []byte) {
		binary.Write(&buf, binary.BigEndian, uint32(len(data)))
		buf.WriteString(ctype)
		buf.Write(data)
		crc := crc32.NewIEEE()
		crc.Write([]byte(ctype))
		crc.Write(data)
		binary.Write(&buf, binary.BigEndian, crc.Sum32())
	}

	// IHDR: 4x4, 8-bit RGB
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:4], 4)  // width
	binary.BigEndian.PutUint32(ihdr[4:8], 4)  // height
	ihdr[8] = 8                                // bit depth
	ihdr[9] = 2                                // color type: RGB
	writeChunk("IHDR", ihdr)

	// IDAT: 4x4 red pixels
	var raw bytes.Buffer
	for y := 0; y < 4; y++ {
		raw.WriteByte(0) // filter: none
		for x := 0; x < 4; x++ {
			raw.Write([]byte{0xff, 0x00, 0x00}) // red
		}
	}
	var zbuf bytes.Buffer
	w, _ := zlib.NewWriterLevel(&zbuf, zlib.BestSpeed)
	w.Write(raw.Bytes())
	w.Close()
	writeChunk("IDAT", zbuf.Bytes())

	// IEND
	writeChunk("IEND", nil)

	return buf.Bytes()
}

func TestProcessChainIdentity(t *testing.T) {
	wasmPath := os.Getenv("RASMCORE_WASM_PATH")
	if wasmPath == "" {
		t.Skip("RASMCORE_WASM_PATH not set; skipping integration test")
	}

	pipe, err := NewPipeline(Options{WASMPath: wasmPath})
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	defer pipe.Close()

	input := makePNG()

	// invert x2 + flip x2 should be identity
	result, err := pipe.Process(input, []Op{
		{Name: "invert"},
		{Name: "invert"},
		{Name: "flip", Params: []float32{0.0}}, // horizontal
		{Name: "flip", Params: []float32{0.0}}, // horizontal
	}, OutputConfig{Format: "png"})
	if err != nil {
		t.Fatalf("Process: %v", err)
	}

	// Result should be valid PNG (starts with signature)
	if len(result) < 8 {
		t.Fatalf("result too short: %d bytes", len(result))
	}
	if !bytes.Equal(result[:4], []byte{0x89, 0x50, 0x4e, 0x47}) {
		t.Fatal("result is not a valid PNG")
	}

	t.Logf("identity chain: %d bytes in → %d bytes out", len(input), len(result))
}

func TestProcessChainBlur(t *testing.T) {
	wasmPath := os.Getenv("RASMCORE_WASM_PATH")
	if wasmPath == "" {
		t.Skip("RASMCORE_WASM_PATH not set; skipping integration test")
	}

	pipe, err := NewPipeline(Options{WASMPath: wasmPath})
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}
	defer pipe.Close()

	input := makePNG()

	result, err := pipe.Process(input, []Op{
		{Name: "blur", Params: []float32{2.0}},
	}, OutputConfig{Format: "png"})
	if err != nil {
		t.Fatalf("Process: %v", err)
	}

	if !bytes.Equal(result[:4], []byte{0x89, 0x50, 0x4e, 0x47}) {
		t.Fatal("result is not a valid PNG")
	}

	t.Logf("blur chain: %d bytes in → %d bytes out", len(input), len(result))
}
