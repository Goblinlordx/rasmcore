# ML Upscale PoC — Real-ESRGAN x4plus

Validates the ML inference execution path before building pipeline integration.
Two self-contained prototypes: native (Rust) and browser (HTML+JS).

## Browser Prototype (fastest to try)

```bash
# Serve the HTML page (any HTTP server works)
npx serve examples/ml-upscale/browser/

# Open http://localhost:3000 in Chrome
# Click "Use Test Image" then "Run Upscale (4x)"
```

The page will:
1. Detect available backends (WebNN, WebGPU, WASM)
2. Download Real-ESRGAN x4plus (~64MB) from HuggingFace
3. Run 4x super-resolution inference
4. Display input and output side-by-side with timing

**WebNN (optional):** Enable `chrome://flags` → "Web Machine Learning Neural Network API"
to test WebNN detection. Functional backend availability varies by platform.

## Native Prototype (Rust)

Requires ONNX Runtime shared library installed on the system.

### macOS (Homebrew)

```bash
brew install onnxruntime
export ORT_DYLIB_PATH=$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib

cd examples/ml-upscale/native
cargo run -- /path/to/input.png /path/to/output.png
```

### macOS (manual)

```bash
# Download from GitHub releases
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-osx-arm64-1.21.0.tgz
tar xzf onnxruntime-osx-arm64-1.21.0.tgz
export ORT_DYLIB_PATH=$(pwd)/onnxruntime-osx-arm64-1.21.0/lib/libonnxruntime.dylib

cd examples/ml-upscale/native
cargo run -- /path/to/input.png /path/to/output.png
```

### Linux

```bash
# Download from GitHub releases (x64 example)
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz
tar xzf onnxruntime-linux-x64-1.21.0.tgz
export ORT_DYLIB_PATH=$(pwd)/onnxruntime-linux-x64-1.21.0/lib/libonnxruntime.so

cd examples/ml-upscale/native
cargo run -- /path/to/input.png /path/to/output.png
```

### Tiled inference (for large images)

```bash
# Use --tile flag to process in 128x128 tiles (avoids GPU memory issues)
cargo run -- large_photo.png upscaled.png --tile 128
```

## What This Validates

- [x] ONNX model download from HuggingFace
- [x] Tensor format conversion: RGBA HWC → RGB NCHW float32
- [x] ONNX Runtime inference produces valid 4x output
- [x] Output conversion: RGB NCHW float32 → RGBA display
- [x] Backend detection: WebNN, WebGPU, WASM
- [x] Tiled inference for large images (native)
- [x] Timing breakdown for each step

## Model

**Real-ESRGAN x4plus** (~64MB, BSD-3-Clause)
- Source: [HuggingFace](https://huggingface.co/qualcomm/Real-ESRGAN-x4plus)
- Input: RGB float32 NCHW [1, 3, H, W], range [0, 1]
- Output: RGB float32 NCHW [1, 3, H*4, W*4], range [0, 1]
