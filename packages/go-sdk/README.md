# rcimg — Go SDK

Image processing pipeline for Go via the rasmcore WASM component.

Uses the `wasmtime` CLI to execute operations — no Component Model host bindings
required. Works with any language that can exec a subprocess.

## Installation

```bash
go get github.com/nicholasgasior/rcimg-go/rcimg
```

## Prerequisites

- [wasmtime CLI](https://wasmtime.dev) installed and in PATH
- rasmcore WASM component built:
  ```bash
  cargo component build -p rasmcore-image --release
  export RASMCORE_WASM_PATH=target/wasm32-wasip1/release/rasmcore_image.wasm
  ```

## Usage

```go
package main

import (
    "log"
    "os"

    "github.com/nicholasgasior/rcimg-go/rcimg"
)

func main() {
    pipe, err := rcimg.NewPipeline(rcimg.Options{})
    if err != nil {
        log.Fatal(err)
    }

    imageBytes, _ := os.ReadFile("input.jpg")

    result, err := pipe.Process(imageBytes, []rcimg.Op{
        {Name: "blur", Params: []float32{5.0}},
        {Name: "sepia", Params: []float32{0.8}},
    }, rcimg.OutputConfig{Format: "jpeg", Quality: ptr(uint8(90))})
    if err != nil {
        log.Fatal(err)
    }

    os.WriteFile("output.jpg", result, 0644)
}

func ptr[T any](v T) *T { return &v }
```

## Available Operations

Use `wasmtime run --invoke 'get-filter-manifest()' rasmcore_image.wasm` to see
all available operations with their parameters.

Common operations:

| Operation | Params | Description |
|-----------|--------|-------------|
| `invert` | (none) | Invert colors |
| `flip` | `[direction]` | 0.0=horizontal, 1.0=vertical |
| `rotate` | `[degrees]` | 90, 180, or 270 |
| `blur` | `[radius]` | Gaussian blur |
| `sharpen` | `[amount]` | Sharpen |
| `sepia` | `[intensity]` | Sepia tone (0.0-1.0) |
| `brightness` | `[amount]` | Brightness adjustment |
| `contrast` | `[amount]` | Contrast adjustment |
| `saturate` | `[factor]` | Saturation (1.0 = no change) |
| `resize` | `[width, height]` | Resize (Lanczos3) |
| `crop` | `[x, y, w, h]` | Crop region |

## How It Works

The SDK uses rasmcore's `process-chain` WIT function which:
1. Accepts a base64-encoded image + list of operations + output format
2. Creates an internal pipeline, applies all operations, encodes the result
3. Returns the output as base64

The Go SDK base64-encodes the input, builds a WAVE invocation string, and
calls `wasmtime run --invoke 'process-chain(...)'` as a subprocess.

## Testing

```bash
cargo component build -p rasmcore-image
RASMCORE_WASM_PATH=target/wasm32-wasip1/debug/rasmcore_image.wasm \
    go test -v ./rcimg/...
```
