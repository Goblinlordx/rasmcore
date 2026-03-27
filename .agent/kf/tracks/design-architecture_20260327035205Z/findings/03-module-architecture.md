# Module Architecture

## Date: 2026-03-27
## Project: rasmcore

---

## Repository Structure

```
rasmcore/
├── core/                          ← THIS REPO (rasmcore/core)
│   ├── wit/                       ← WIT interface definitions (the contracts)
│   │   ├── core/
│   │   │   ├── types.wit
│   │   │   └── errors.wit
│   │   ├── image/
│   │   │   ├── decoder.wit
│   │   │   ├── encoder.wit
│   │   │   ├── transform.wit
│   │   │   └── filters.wit
│   │   ├── video/
│   │   │   ├── demuxer.wit
│   │   │   └── muxer.wit
│   │   ├── audio/
│   │   │   └── transform.wit
│   │   ├── data/
│   │   │   ├── converter.wit
│   │   │   └── table.wit
│   │   └── codec/
│   │       ├── encoder.wit
│   │       └── decoder.wit
│   │
│   ├── crates/                    ← Rust implementations
│   │   ├── rasmcore-types/        ← Shared Rust types (generated from WIT)
│   │   ├── rasmcore-image/        ← Image processing component
│   │   ├── rasmcore-video/        ← Video container handling component
│   │   ├── rasmcore-audio/        ← Audio processing component
│   │   ├── rasmcore-data/         ← Data processing component
│   │   ├── rasmcore-codec-av1/    ← AV1 codec plugin (open, bundled)
│   │   ├── rasmcore-codec-vp9/    ← VP9 codec plugin (open, bundled)
│   │   ├── rasmcore-codec-flac/   ← FLAC codec plugin (open, bundled)
│   │   └── rasmcore-codec-opus/   ← Opus codec plugin (open, bundled)
│   │
│   ├── tests/                     ← Integration, parity, and benchmark tests
│   │   ├── parity/                ← Output comparison vs native tools
│   │   ├── benchmarks/            ← WASM vs native performance
│   │   └── fixtures/              ← Test media files
│   │
│   ├── Cargo.toml                 ← Workspace manifest
│   └── .agent/kf/                 ← Kiloforge project management
│
├── non-free/                      ← SEPARATE REPO (rasmcore/non-free)
│   ├── crates/
│   │   ├── rasmcore-codec-h264/   ← H.264 codec (patent-encumbered)
│   │   ├── rasmcore-codec-h265/   ← H.265/HEVC codec (patent-encumbered)
│   │   └── rasmcore-codec-aac/    ← AAC codec (patent-encumbered)
│   └── wit/                       ← Uses same rasmcore:codec interfaces
│
└── sdk/                           ← FUTURE REPO (rasmcore/sdk)
    ├── ts/                        ← TypeScript SDK (jco-transpiled + ergonomic wrappers)
    └── examples/                  ← Usage examples per language
```

---

## Module Dependency Graph

```
rasmcore:core/types ← (used by all)
       │
       ├── rasmcore:image/{decoder,encoder,transform,filters}
       │
       ├── rasmcore:codec/{encoder,decoder}  ← Plugin interface
       │     │
       │     ├── rasmcore-codec-av1    (implements rasmcore:codec)
       │     ├── rasmcore-codec-vp9    (implements rasmcore:codec)
       │     ├── rasmcore-codec-flac   (implements rasmcore:codec)
       │     ├── rasmcore-codec-opus   (implements rasmcore:codec)
       │     ├── rasmcore-codec-h264   (implements rasmcore:codec) [non-free]
       │     └── rasmcore-codec-h265   (implements rasmcore:codec) [non-free]
       │
       ├── rasmcore:video/{demuxer,muxer}
       │     └── imports rasmcore:codec/{encoder,decoder}
       │
       ├── rasmcore:audio/transform
       │
       └── rasmcore:data/{converter,table}
```

**Key architectural rule:** All codec implementations export the same `rasmcore:codec` interfaces. The video module imports these interfaces — it doesn't know or care which codec plugin is wired in. Composition happens at build time via `wasm-tools compose`.

---

## Component Composition Examples

### Minimal: Image Processing Only

```bash
# Just image processing — no video, no codecs
rasmcore-image.wasm
# User imports this single component
```

### Standard: Video with AV1

```bash
# Compose video processor with AV1 codec
wasm-tools compose rasmcore-video.wasm \
  -d rasmcore-codec-av1.wasm \
  -o video-av1.wasm
```

### Full: Video with Multiple Codecs

```bash
# Compose with multiple codecs
wasm-tools compose rasmcore-video.wasm \
  -d rasmcore-codec-av1.wasm \
  -d rasmcore-codec-h264.wasm \   # from non-free repo
  -o video-full.wasm
```

### Custom: User Picks What They Need

```bash
# User composes exactly what they need
wasm-tools compose rasmcore-image.wasm \
  -d rasmcore-video.wasm \
  -d rasmcore-codec-av1.wasm \
  -d rasmcore-data.wasm \
  -o my-media-toolkit.wasm
```

---

## Crate Design Pattern

Each crate follows the same structure:

```
rasmcore-image/
├── Cargo.toml
├── wit/                    ← WIT deps (symlinks or copies from root wit/)
│   └── deps/
├── src/
│   ├── lib.rs             ← WIT binding glue + module registration
│   ├── decoder.rs         ← Implements rasmcore:image/decoder
│   ├── encoder.rs         ← Implements rasmcore:image/encoder
│   ├── transform.rs       ← Implements rasmcore:image/transform
│   └── filters.rs         ← Implements rasmcore:image/filters
└── tests/
    ├── parity/            ← Compare output vs libpng, libjpeg, etc.
    └── bench/             ← WASM vs native benchmarks
```

### Cargo.toml Pattern

```toml
[package]
name = "rasmcore-image"
version = "0.1.0"
edition = "2024"

[dependencies]
# Pure Rust image library — WASM-ready
image = { version = "0.25", default-features = false, features = [
    "png", "jpeg", "gif", "webp", "bmp", "tiff", "avif", "ico", "qoi"
] }
photon-rs = "0.3"

[lib]
crate-type = ["cdylib"]

[package.metadata.component]
package = "rasmcore:image"
```

---

## Build System

### Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "crates/rasmcore-types",
    "crates/rasmcore-image",
    "crates/rasmcore-video",
    "crates/rasmcore-audio",
    "crates/rasmcore-data",
    "crates/rasmcore-codec-av1",
    "crates/rasmcore-codec-vp9",
    "crates/rasmcore-codec-flac",
    "crates/rasmcore-codec-opus",
]

[workspace.dependencies]
image = { version = "0.25", default-features = false }
photon-rs = "0.3"
```

### Build Commands

```bash
# Build all components
cargo component build --release --workspace

# Build single component
cargo component build --release -p rasmcore-image

# Validate component
wasm-tools validate target/wasm32-wasip2/release/rasmcore_image.wasm

# Compose components
wasm-tools compose \
  target/wasm32-wasip2/release/rasmcore_video.wasm \
  -d target/wasm32-wasip2/release/rasmcore_codec_av1.wasm \
  -o composed-video.wasm
```

---

## Testing Architecture

### Parity Tests

For each module, compare output against reference implementations:

| Module | Reference Tool | Comparison |
|--------|---------------|------------|
| Image decode (PNG) | libpng | Pixel-perfect output match |
| Image decode (JPEG) | libjpeg-turbo | PSNR > 60dB (lossy comparison) |
| Image resize | ImageMagick | SSIM > 0.99 |
| AV1 encode | rav1e (native) | Bitstream identical (same params) |
| AV1 decode | rav1d (native) | Frame-perfect match |
| CSV parse | Python pandas | Row/column exact match |
| Parquet read | Apache Arrow | Schema + data exact match |

### Benchmark Tests

```bash
# Run benchmark: WASM vs native
cargo bench --target wasm32-wasip2
cargo bench  # native comparison

# Output: table of operation × time × overhead %
```

Benchmark categories:
- **Throughput**: MB/s for format conversion
- **Latency**: Time per operation (encode frame, decode image, etc.)
- **Memory**: Peak memory usage per operation
- **Overhead**: WASM time / native time ratio

---

## Host SDK API Surface

### TypeScript (via jco)

```typescript
// jco transpiles rasmcore-image.wasm to ES module
import { decoder, encoder, transform, filters } from '@rasmcore/image';

// Decode an image
const result = decoder.decode(pngBytes);
if (result.tag === 'ok') {
  const { pixels, info } = result.val;
  console.log(`${info.width}x${info.height} ${info.format}`);
}

// Resize
const resized = transform.resize(pixels, info, 800, 600, 'lanczos3');

// Encode to JPEG
const jpeg = encoder.encode(resized.val[0], resized.val[1], 'jpeg', 85);
```

### Rust (via wasmtime)

```rust
use wasmtime::component::*;

// Load and instantiate the image component
let component = Component::from_file(&engine, "rasmcore-image.wasm")?;
let instance = linker.instantiate(&mut store, &component)?;

// Call decode
let result = instance.call_decode(&mut store, &png_bytes)?;
```

### Go (future — when Component Model support arrives)

Go consumers will need to wait for wasmtime-go Component Model support, or use rasmcore via CLI/HTTP wrapper.
