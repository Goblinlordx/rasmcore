# rasmcore — Python SDK

Pure-Rust WASM image processing pipeline with optional GPU acceleration.

## Installation

```bash
pip install rasmcore            # CPU only
pip install rasmcore[gpu]       # With GPU support (wgpu)
```

## Usage

```python
from rasmcore import Pipeline

# Create pipeline (auto-detects GPU)
pipe = Pipeline()

# Load image
with open("input.jpg", "rb") as f:
    img = pipe.read(f.read())

# Apply operations
blurred = pipe.blur(img, radius=10.0)
toned = pipe.sepia(blurred, intensity=0.5)

# Export
result = pipe.write_jpeg(toned, quality=90)
with open("output.jpg", "wb") as f:
    f.write(result)
```

## GPU Acceleration

When installed with `[gpu]`, rasmcore automatically uses GPU compute
shaders for expensive operations (blur, distortion, bilateral, etc.).
GPU is auto-detected; no configuration needed.

To disable GPU for a specific write:
```python
result = pipe.write_png(node)  # Uses GPU if available
```

To disable GPU entirely:
```python
pipe = Pipeline(use_gpu=False)
```

## Building from source

```bash
# Build WASM component
cargo component build -p rasmcore-image --release

# Install Python SDK in development mode
cd packages/python-sdk
pip install -e ".[gpu]"

# Set WASM path
export RASMCORE_WASM_PATH=../../target/wasm32-wasip1/release/rasmcore_image.wasm
```
