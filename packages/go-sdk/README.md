# rcimg — Go SDK

Pure-Rust WASM image processing pipeline for Go with optional GPU acceleration.

## Installation

```bash
go get github.com/nicholasgasior/rcimg-go/rcimg
```

For GPU support (requires wgpu-native):
```bash
go get github.com/nicholasgasior/rcimg-go/gpu
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
    defer pipe.Close()

    imageBytes, _ := os.ReadFile("input.jpg")

    img, err := pipe.Read(imageBytes)
    if err != nil {
        log.Fatal(err)
    }

    blurred, err := pipe.Blur(img, rcimg.BlurConfig{Radius: 10.0})
    if err != nil {
        log.Fatal(err)
    }

    result, err := pipe.WriteJPEG(blurred, rcimg.JPEGConfig{Quality: 90})
    if err != nil {
        log.Fatal(err)
    }

    os.WriteFile("output.jpg", result, 0644)
}
```

## GPU Acceleration

Build with the `gpu` tag and wgpu-native installed:

```bash
go build -tags gpu .
```

```go
import "github.com/nicholasgasior/rcimg-go/gpu"

executor, err := gpu.New()
if err != nil {
    log.Println("No GPU, falling back to CPU:", err)
}

pipe, _ := rcimg.NewPipeline(rcimg.Options{
    UseGPU:      true,
    GpuExecutor: executor,
})
```

## Status

**Note:** The WASM Component Model is not yet fully supported by wazero.
The pipeline methods currently return placeholder errors. Full support
is expected when wazero adds Component Model support, or via wasmtime-go
CGO bindings as an alternative.

## Building the WASM component

```bash
cargo component build -p rasmcore-image --release
export RASMCORE_WASM_PATH=target/wasm32-wasip1/release/rasmcore_image.wasm
```
