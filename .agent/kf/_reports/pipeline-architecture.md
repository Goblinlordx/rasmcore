# Demand-Driven Tile Pipeline Architecture

> Design document for rasmcore-image pipeline architecture.
> Updated: 2026-03-28

## 1. Overview

rasmcore-image shifts from stateless buffer-in/buffer-out functions to a **demand-driven (pull-based) tile pipeline**. The key principle: `write()` drives execution by pulling tiles backward through a graph of image-node resources. Each node requests tiles from its upstream with appropriate overlap for its kernel. Only active tiles occupy memory.

### Goals

1. **Minimal memory** — Only tiles currently being processed occupy memory
2. **Streaming I/O** — Decoders produce tiles on demand, encoders consume them
3. **Operation fusion** — Pipeline sees full operation chain before executing (e.g., read+resize = shrink-on-load)
4. **Chainable API** — `pipeline.read(data).resize(800,600).write('jpeg', config)`
5. **Backward compatible** — Existing stateless functions become convenience wrappers

### Public API Naming

| Current | New | Rationale |
|---------|-----|-----------|
| decode | read | Users think "read a JPEG", not "decode" |
| encode | write | Users think "write a PNG", not "encode" |
| decode_as | read_as | Consistent with read |

Internal domain code retains decode/encode since those are technically accurate at the implementation level. The rename is at the WIT/adapter boundary only.

---

## 2. WIT Interface Design

### 2.1 Image Node Resource

The `image-node` is the core abstraction. Every operation produces an image-node. Nodes are lazy — they don't compute pixels until a tile is requested.

```wit
package rasmcore:image@0.2.0;

interface pipeline {
    use rasmcore:core/types.{buffer, image-info, pixel-format};
    use rasmcore:core/errors.{rasmcore-error};

    /// A tile of pixel data at a specific position.
    record tile {
        pixels: buffer,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        info: image-info,
    }

    /// A lazy image node — pixels computed on demand.
    /// Created by read() or transform operations.
    resource image-node {
        /// Get the image dimensions and format (available without computing pixels).
        info: func() -> image-info;

        /// Request a tile from this node. The node computes it by pulling
        /// from upstream nodes as needed.
        get-tile: func(x: u32, y: u32, width: u32, height: u32) -> result<tile, rasmcore-error>;

        /// Materialize the full image (convenience for small images or
        /// operations that need the whole thing).
        materialize: func() -> result<tile, rasmcore-error>;
    }

    // ─── Source Nodes ───

    /// Create a source node from encoded image data.
    /// Does NOT decode — only parses headers for dimensions/format.
    read: func(data: buffer) -> result<image-node, rasmcore-error>;

    /// Detect the format of encoded image data.
    detect-format: func(header: buffer) -> option<string>;

    /// List supported read formats.
    supported-read-formats: func() -> list<string>;

    // ─── Transform Nodes ───
    // Each wraps an upstream node, producing a new node.

    /// Resize filter algorithms.
    enum resize-filter { nearest, bilinear, bicubic, lanczos3 }

    /// Rotation amounts.
    enum rotation { r90, r180, r270 }

    /// Flip direction.
    enum flip-direction { horizontal, vertical }

    /// Resize to exact dimensions.
    resize: func(source: image-node, width: u32, height: u32, filter: resize-filter) -> image-node;

    /// Crop a region.
    crop: func(source: image-node, x: u32, y: u32, width: u32, height: u32) -> result<image-node, rasmcore-error>;

    /// Rotate by fixed angle.
    rotate: func(source: image-node, angle: rotation) -> image-node;

    /// Flip horizontally or vertically.
    flip: func(source: image-node, direction: flip-direction) -> image-node;

    /// Convert pixel format.
    convert-format: func(source: image-node, target: pixel-format) -> image-node;

    // ─── Filter Nodes ───

    /// Gaussian blur.
    blur: func(source: image-node, radius: f32) -> result<image-node, rasmcore-error>;

    /// Sharpen (unsharp mask).
    sharpen: func(source: image-node, amount: f32) -> image-node;

    /// Brightness adjustment (-1.0 to 1.0).
    brightness: func(source: image-node, amount: f32) -> result<image-node, rasmcore-error>;

    /// Contrast adjustment (-1.0 to 1.0).
    contrast: func(source: image-node, amount: f32) -> result<image-node, rasmcore-error>;

    /// Convert to grayscale.
    grayscale: func(source: image-node) -> image-node;

    // ─── Sink (Write) ───

    /// JPEG write configuration.
    record jpeg-write-config {
        quality: option<u8>,
    }

    /// PNG write configuration.
    record png-write-config {
        compression-level: option<u8>,
    }

    /// WebP write configuration.
    record webp-write-config {
        quality: option<u8>,
        lossless: option<bool>,
    }

    /// Write node to JPEG. Pulls all tiles through the chain.
    write-jpeg: func(source: image-node, config: jpeg-write-config) -> result<buffer, rasmcore-error>;

    /// Write node to PNG.
    write-png: func(source: image-node, config: png-write-config) -> result<buffer, rasmcore-error>;

    /// Write node to WebP.
    write-webp: func(source: image-node, config: webp-write-config) -> result<buffer, rasmcore-error>;

    /// Write node to any format (convenience).
    write: func(source: image-node, format: string, quality: option<u8>) -> result<buffer, rasmcore-error>;

    /// List supported write formats.
    supported-write-formats: func() -> list<string>;
}
```

### 2.2 World Definition

```wit
world image-processor {
    export pipeline;

    // Legacy stateless exports preserved for backward compatibility
    export decoder;   // wraps pipeline.read + materialize
    export encoder;   // wraps pipeline.write
    export transform; // wraps pipeline transform nodes
    export filters;   // wraps pipeline filter nodes
}
```

### 2.3 Host SDK Experience

**TypeScript (via jco):**
```typescript
import { pipeline } from 'rasmcore-image';

// Chainable via node returns
const source = pipeline.read(inputBuffer);
const resized = pipeline.resize(source, 800, 600, 'lanczos3');
const sharpened = pipeline.sharpen(resized, 1.0);
const output = pipeline.writeJpeg(sharpened, { quality: 85 });
```

**Rust (via wasmtime):**
```rust
let source = pipeline.call_read(&mut store, &input_data)?;
let resized = pipeline.call_resize(&mut store, source, 800, 600, ResizeFilter::Lanczos3)?;
let output = pipeline.call_write_jpeg(&mut store, resized, &config)?;
```

**Future host-side wrapper** (ergonomic builder, not in WIT):
```typescript
// SDK-level sugar built on top of the pipeline interface
const output = await rasmcore.image()
  .read(inputBuffer)
  .resize(800, 600, 'lanczos3')
  .sharpen(1.0)
  .writeJpeg({ quality: 85 });
```

---

## 3. Internal Node Architecture

### 3.1 Node Trait (Rust Domain)

```rust
/// The core trait every pipeline node implements.
trait ImageNode {
    /// Image dimensions and format (available without computing pixels).
    fn info(&self) -> &ImageInfo;

    /// Compute a tile. The node may request tiles from upstream.
    fn get_tile(&self, request: TileRequest) -> Result<Tile, ImageError>;

    /// Overlap this node needs from upstream for each output tile.
    fn overlap(&self) -> Overlap;
}

struct TileRequest {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

struct Tile {
    pixels: Vec<u8>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    info: ImageInfo,
}

/// How much extra context a node needs from upstream.
struct Overlap {
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
}
```

### 3.2 Tile Request Propagation

When `write()` is called, it iterates output tiles and each request cascades:

```
write_jpeg(node, config)
  for each output tile (x, y, w, h):
    tile = node.get_tile(x, y, w, h)
      ↓
    sharpen.get_tile(x, y, w, h)
      needs overlap: top=1, bottom=1, left=1, right=1
      requests upstream: get_tile(x-1, y-1, w+2, h+2)
      ↓
    resize.get_tile(x', y', w', h')     [coordinates mapped to source scale]
      needs overlap: depends on filter (lanczos3 = 3px each side)
      requests upstream: get_tile(x'', y'', w'', h'')
      ↓
    source_decoder.get_tile(x'', y'', w'', h'')
      decodes the required region (or full image + caches)
```

### 3.3 Overlap Calculation Per Operation

| Operation | Overlap | Notes |
|-----------|---------|-------|
| Point ops (brightness, contrast, invert) | 0 | Each pixel independent |
| Blur (radius r) | ceil(3*sigma) each side | Gaussian kernel extent |
| Sharpen (unsharp mask) | 1-3 each side | Depends on kernel size |
| Resize | ceil(filter_radius * scale) | Lanczos3 = 3 lobes |
| Crop | 0 | Just coordinate offset |
| Rotate 90/180/270 | 0 (remaps coordinates) | Tile coords transform |
| Flip | 0 (remaps coordinates) | Tile coords mirror |
| Grayscale | 0 | Per-pixel channel combine |
| Edge detect (future) | 1-2 | 3x3 or 5x5 kernel |
| Median filter (future) | radius | Window size |
| Custom convolution | kernel_size/2 | Arbitrary kernel |

### 3.4 Operation Fusion

The pipeline can detect optimization opportunities before executing:

| Pattern | Fusion | Savings |
|---------|--------|---------|
| read(JPEG) + resize(1/N) | JPEG DCT shrink-on-load | Decode at 1/2, 1/4, or 1/8 size |
| read(TIFF) + crop | Decode only the tiled region | Skip unused tiles |
| brightness + contrast | Single linear transform: `a * pixel + b` | One pass instead of two |
| multiple point ops | Compose into single LUT | One lookup per pixel |
| resize + resize | Compose into single resize | One resampling pass |

**Fusion detection:** After the node chain is built (before first `get_tile`), a fusion pass walks the chain looking for known patterns. Fused nodes replace the original chain segment.

### 3.5 Tile Size Strategy

**Default tile size: full-width strips of 64 rows.**

Rationale:
- Full-width strips align with JPEG MCU rows and PNG scanline decompression
- 64 rows is a good balance between cache efficiency and overhead
- For a 4000px wide RGBA image: 64 * 4000 * 4 = 1MB per strip — fits in L2/L3 cache
- The write() sink iterates top-to-bottom in strip order

For formats with native tiles (TIFF), use the format's tile size instead.

---

## 4. Format Tile Access Analysis

| Format | Access Type | Details | Seek Cache Needed |
|--------|------------|---------|-------------------|
| **JPEG** | PARTIAL | Restart markers every N MCU rows allow seeking to row boundaries. DCT shrink-on-load (1/2, 1/4, 1/8). Without restart markers, sequential only. | Yes, sparse (between restart markers) |
| **PNG** | SEQUENTIAL | IDAT chunks form a single zlib stream. Must decompress from start. `png` crate supports row-by-row reading via `next_row()`. | Yes, checkpoint every N rows |
| **TIFF (tiled)** | RANDOM | Native tile offsets in IFD. Each tile independently compressed. Perfect random access. | No |
| **TIFF (stripped)** | PARTIAL | Strip offsets in IFD. Each strip independently compressed. Random access per strip. | No (strips are directly addressable) |
| **WebP (lossy)** | SEQUENTIAL | VP8 bitstream is sequential. No random access to regions. | Yes, checkpoint every N rows |
| **WebP (lossless)** | SEQUENTIAL | VP8L bitstream is sequential. | Yes |
| **GIF** | SEQUENTIAL | LZW compressed frame data is sequential. | Yes (but images are typically small) |
| **AVIF** | PARTIAL | AV1 supports tiles (configurable grid). Each tile can be independently decoded. | Depends on encoder settings |
| **BMP** | RANDOM | Uncompressed: direct offset calculation. RLE: sequential. | No (uncompressed) / Yes (RLE) |
| **QOI** | SEQUENTIAL | Streaming chunk format, no random access. | Yes |

### 4.1 Sparse Seek Cache Design

For sequential formats, the decoder node maintains a **checkpoint cache**:

```rust
struct SeekCache {
    /// Decoded scanlines cached at regular intervals.
    checkpoints: BTreeMap<u32, Vec<u8>>,  // row -> full scanline pixels
    /// Decoder state checkpoint (if format supports it).
    decoder_states: BTreeMap<u32, DecoderState>,
    /// Interval between checkpoints (in rows).
    interval: u32,
    /// Maximum number of cached checkpoints before eviction.
    max_checkpoints: usize,
}
```

**How it works:**
1. First forward pass: every `interval` rows, cache the decoded scanline and decoder state
2. Backward seek to row R: find nearest checkpoint at or before R, restore decoder state, decode forward to R
3. Eviction: LRU eviction when `max_checkpoints` exceeded, keeping checkpoints near recent access patterns

**Default parameters:**
- `interval`: 64 rows (matches tile strip height)
- `max_checkpoints`: 16 (covers ~1024 rows with 64-row interval)
- Memory cost: 16 * width * bytes_per_pixel per format. For 4000px RGBA: 16 * 4000 * 4 = 256KB

**Sequential-only optimization:** If the pipeline detects strictly top-to-bottom access (no backward seeks), skip caching entirely — just stream forward. This is the common case for `read → transform → write` pipelines.

---

## 5. Migration Plan

### Phase 1: Core Pipeline Engine (No WIT Changes)

Build the internal node graph engine in the domain layer:
- `ImageNode` trait
- Source node (wraps current decoder)
- Transform nodes (resize, crop, rotate, flip)
- Filter nodes (blur, sharpen, brightness, contrast, grayscale)
- Sink (write) that iterates tiles through the chain
- Tile request propagation with overlap
- **Test:** Pipeline produces identical output to current stateless functions

### Phase 2: WIT Pipeline Interface

Add the new `pipeline` WIT interface:
- `image-node` resource
- `read()`, transform functions, `write()` functions
- Adapter layer maps WIT resources to domain nodes
- **Test:** WASM integration tests via wasmtime, TS SDK tests via jco

### Phase 3: Operation Fusion

Add the fusion optimization pass:
- JPEG shrink-on-load (read + resize)
- Point operation composition (brightness + contrast → single linear)
- **Test:** Fused pipeline produces identical output, benchmark shows speedup

### Phase 4: Seek Cache for Sequential Formats

Add the sparse seek cache for PNG, WebP, GIF, QOI:
- Checkpoint cache with configurable interval
- LRU eviction
- Skip caching for forward-only access patterns
- **Test:** Random tile access on PNG produces correct results

### Phase 5: Deprecate Stateless Interface

- Stateless `decoder`/`encoder`/`transform`/`filters` interfaces become thin wrappers around pipeline
- Mark as deprecated in WIT comments
- No removal — backward compatibility forever at 0.x

### Phase 6: WASI-IO Stream Integration

- `read-from-stream(source: input-stream) -> result<image-node, error>`
- `write-to-stream(source: image-node, format, config, dest: output-stream) -> result<_, error>`
- Decoder streams encoded bytes in chunks, produces tiles on demand
- Encoder consumes tiles, streams encoded bytes out

---

## 6. Memory Model Analysis

### Current: Full Image in Memory

| Image Size | Pixels (RGBA) | Peak Memory |
|------------|---------------|-------------|
| 1 MP (1000x1000) | 4 MB | ~12 MB (input + decoded + output) |
| 10 MP (3648x2736) | 40 MB | ~120 MB |
| 100 MP (10000x10000) | 400 MB | ~1.2 GB |

Every operation materializes a full copy. A 3-step pipeline (read → resize → write) holds 3 full copies simultaneously.

### Tile Pipeline: Active Tiles Only

| Image Size | Tile Strip (64 rows) | Peak Memory (3-step pipeline) |
|------------|---------------------|-------------------------------|
| 1 MP (1000x1000) | 256 KB | ~2 MB (3 strips in flight + overhead) |
| 10 MP (3648x2736) | 936 KB | ~5 MB |
| 100 MP (10000x10000) | 2.5 MB | ~12 MB |

With seek cache add ~256KB for checkpoint storage. **50-100x memory reduction** for large images.

### Cross-Boundary Copy Reduction

| Scenario | Current (stateless) | Pipeline |
|----------|-------------------|----------|
| read → resize → sharpen → write | 4 full-image copies across host/guest boundary | 1 copy in (encoded bytes) + 1 copy out (encoded bytes) |
| read → crop(small region) → write | Full decode + full copy + encode | Decode only crop region + encode |

---

## 7. Key Design Decisions

### D1: image-node is a WIT resource (not opaque handle)
Resources give hosts typed methods (`.info()`, `.getTile()`) and automatic cleanup via drop. Handles would require manual lifecycle management.

### D2: Transform functions are module-level, not resource methods
`resize(source, w, h, filter)` not `source.resize(w, h, filter)`. This matches WIT convention where resources are data, functions are operations. It also allows the pipeline to see the full chain before execution (fusion).

### D3: Tile size is full-width strips
Simplifies coordinate math, aligns with format scanline structure, good cache behavior. Tiled TIFF is the exception (uses format tiles).

### D4: Fusion is optional and transparent
The pipeline always produces correct results without fusion. Fusion is a performance optimization detected automatically. New fusion rules can be added without API changes.

### D5: Backward compatibility via wrapper interfaces
The existing `decoder`/`encoder`/`transform`/`filters` interfaces remain forever. They internally create a one-step pipeline. Zero breaking changes for current consumers.

### D6: read/write naming at WIT boundary only
Domain code keeps decode/encode terminology. The rename happens in the WIT interface and adapter layer. This is a clean separation — domain is technically precise, WIT is user-friendly.

### D7: Seek cache is per-source-node, not global
Each source node manages its own seek cache. This avoids shared state and makes nodes independently testable. Memory budget is per-node.

---

## 8. Open Questions

### Q1: Should image-node.get_tile() be public in WIT?
**Recommendation: Yes, but secondary.** Most users use `write()` which drives tiles internally. Power users (custom renderers, tiled output) benefit from direct tile access. Expose it but document `write()` as the primary API.

### Q2: Thread-safe parallel tile computation?
**Recommendation: Defer.** WIT resources are single-threaded per instance. The host could create multiple pipeline instances for parallelism. Internal Rust code could use rayon for parallel tile computation within a single `get_tile()` call (e.g., parallel scanlines within a strip). Design for it, don't implement yet.

### Q3: Lazy node chain vs immediate execution?
**Recommendation: Lazy with explicit materialize.** Nodes queue their parameters. First `get_tile()` or `write()` triggers actual computation. This enables fusion. `materialize()` is the escape hatch for operations that truly need the full image.

### Q4: How to handle global operations (histogram equalize)?
**Recommendation: Two-pass node.** First `get_tile()` triggers a full statistics pass (iterates all upstream tiles to compute histogram). Results are cached. Subsequent `get_tile()` calls apply the equalization using cached stats. The statistics pass is the cost — it's unavoidable for global operations.
