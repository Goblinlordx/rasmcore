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

### 3.1 Pipeline-Owned Node Graph with Tile Pool

**Critical design revision:** Transform functions do NOT take ownership of source
nodes. A source node must persist across multiple tile requests (e.g., blur on
tile (0,0) and blur on tile (1,0) both need overlapping pixels from the same
upstream). WIT `borrow<>` is call-scoped only, so it can't persist either.

**Solution: The pipeline resource owns all nodes internally.** Transform functions
accept and return opaque `node-id` values (u32). The pipeline holds a
`Vec<Box<dyn ImageNode>>` and a shared tile pool.

#### Revised WIT (pipeline as graph owner)

```wit
resource pipeline {
    constructor();

    /// Opaque handle to a node in the pipeline's internal graph.
    /// The pipeline owns all nodes — node-ids are indices, not resources.

    read: func(data: buffer) -> result<node-id, rasmcore-error>;
    resize: func(source: node-id, w: u32, h: u32, filter: resize-filter) -> node-id;
    crop: func(source: node-id, x: u32, y: u32, w: u32, h: u32) -> result<node-id, rasmcore-error>;
    rotate: func(source: node-id, angle: rotation) -> node-id;
    flip: func(source: node-id, direction: flip-direction) -> node-id;
    blur: func(source: node-id, radius: f32) -> result<node-id, rasmcore-error>;
    sharpen: func(source: node-id, amount: f32) -> node-id;
    brightness: func(source: node-id, amount: f32) -> result<node-id, rasmcore-error>;
    contrast: func(source: node-id, amount: f32) -> result<node-id, rasmcore-error>;
    grayscale: func(source: node-id) -> node-id;

    /// Node info (available without computing pixels).
    node-info: func(node: node-id) -> result<image-info, rasmcore-error>;

    /// Drive execution — pulls tiles through the graph.
    write-jpeg: func(source: node-id, config: jpeg-write-config) -> result<buffer, rasmcore-error>;
    write-png: func(source: node-id, config: png-write-config) -> result<buffer, rasmcore-error>;
    write-webp: func(source: node-id, config: webp-write-config) -> result<buffer, rasmcore-error>;
    write: func(source: node-id, format: string, quality: option<u8>) -> result<buffer, rasmcore-error>;
}

type node-id = u32;
```

This enables **DAG topologies** — the same source node can feed multiple transforms:
```typescript
const p = new Pipeline();
const src = p.read(data);              // node 0
const thumb = p.resize(src, 200, 200); // node 1, refs node 0
const large = p.resize(src, 2000, 2000); // node 2, also refs node 0
```

### 3.2 Spatial Tile Cache with Reference-Counted Borrowing

Tiles are NOT owned by nodes. They're owned by a **pipeline-level spatial cache**.
Nodes request arbitrary rectangular regions. The cache:
1. Checks if the requested region (or parts of it) is already cached
2. Computes only the missing sub-regions from upstream
3. Assembles the full requested region from cached + new fragments
4. Returns a ref-counted handle; slot is reclaimable when rc hits 0

Since execution is single-threaded, no locks needed.

**Key principle: dynamically-sized region queries, not fixed tiles.** Each node
requests exactly the region it needs — no alignment to fixed grid boundaries.
The write sink's output format determines the initial chunk geometry (JPEG: MCU
rows, PNG: scanlines, TIFF: native tiles), and each upstream node expands the
request by its kernel overlap. Every rectangle in the chain is precisely sized.

```rust
/// A rectangular region in pixel coordinates.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// A cached pixel region for a specific node.
struct CachedRegion {
    rect: Rect,
    pixels: Vec<u8>,      // pixel data for this region
    rc: u32,              // reference count (0 = reclaimable)
    generation: u32,      // incremented on reuse, prevents stale handles
    node_id: u32,         // which node produced this region
}

/// Handle returned to nodes — lightweight, copyable.
#[derive(Clone, Copy)]
struct RegionHandle {
    index: u32,
    generation: u32,
}

/// Pipeline-owned spatial cache.
struct SpatialCache {
    /// All cached regions, indexed by slot.
    regions: Vec<CachedRegion>,
    /// Spatial index per node: node_id → list of region indices.
    /// For common sequential access, a simple sorted Vec<(Rect, usize)>
    /// suffices. R-tree for complex 2D access patterns (future).
    node_index: HashMap<u32, Vec<usize>>,
    /// Memory budget (max total pixel bytes across all regions).
    memory_budget: usize,
    memory_used: usize,
}

impl SpatialCache {
    /// Request a region from a node. Returns cached data if available,
    /// computing only the missing sub-regions from upstream.
    fn acquire(
        &mut self,
        node_id: u32,
        request: Rect,
        compute: impl FnOnce(Rect, &mut Vec<u8>),
    ) -> RegionHandle {
        // 1. Find cached regions for this node that intersect the request
        let (covered, missing) = self.spatial_query(node_id, request);

        if missing.is_empty() {
            // Full cache hit — find the containing region, increment rc
            return self.ref_existing(node_id, request);
        }

        if covered.is_empty() {
            // Full cache miss — compute the entire region
            let slot = self.alloc_slot(request.pixel_bytes());
            self.regions[slot].node_id = node_id;
            self.regions[slot].rect = request;
            self.regions[slot].rc = 1;
            compute(request, &mut self.regions[slot].pixels);
            self.index_region(node_id, slot);
            return self.handle_for(slot);
        }

        // Partial hit — compute only missing fragments, assemble
        let slot = self.alloc_slot(request.pixel_bytes());
        self.assemble_from_cached_and_computed(
            slot, node_id, request, &covered, &missing, compute
        );
        self.handle_for(slot)
    }

    /// Compute rectangular difference: request minus cached regions.
    /// Returns (covered_rects, missing_rects) where missing_rects are
    /// non-overlapping axis-aligned rectangles covering the uncovered area.
    fn spatial_query(&self, node_id: u32, request: Rect) -> (Vec<Rect>, Vec<Rect>) {
        // For sequential (1D) access: simple range difference
        // For 2D access: axis-aligned rectangle clipping
        // ...
    }

    /// Release a region handle. Decrements rc.
    fn release(&mut self, handle: RegionHandle) {
        let region = &mut self.regions[handle.index as usize];
        assert_eq!(region.generation, handle.generation);
        region.rc -= 1;
        // When rc=0, region is reclaimable but stays in cache for potential reuse.
        // Evicted only when memory budget is exceeded.
    }

    /// Read region pixels.
    fn read(&self, handle: RegionHandle) -> &[u8] {
        let region = &self.regions[handle.index as usize];
        assert_eq!(region.generation, handle.generation);
        &region.pixels
    }
}
```

**Why dynamically-sized queries are better than fixed tiles:**
- No wasted pixels — each node requests exactly what it needs
- No alignment constraints — no need to pick tile size a priori
- Overlap reuse is automatic — adjacent output regions that overlap in their
  upstream requests share cached pixels without redundant computation
- Write sink format determines chunk geometry naturally (JPEG MCU rows, PNG
  scanlines, TIFF native tiles) — no conflict with a fixed tile grid

**Tile identity is (node_id, rect).** Region at (0,0,20,20) from the source
decoder (node 0) is a different cache entry than (0,0,20,20) from the blur node
(node 2). Each node's output regions are independently cached.

#### Example: blur pipeline with overlap reuse

```
write_jpeg iterates MCU rows (8 rows each, full width):

Output row 0 (y=0, h=8):
  blur (radius=5) needs upstream region (y=-5..13, padded to 0..13)
    cache.acquire(node:source, rect:(0,0,W,13))
    → cache miss, compute: decode rows 0-13
    → cached, rc=1
  blur produces output (0,0,W,8) from source (0,0,W,13)
  encoder consumes output
  blur releases — but source (0,0,W,13) stays cached (rc=0 but not evicted)

Output row 1 (y=8, h=8):
  blur needs upstream region (y=3..21)
    cache.acquire(node:source, rect:(0,3,W,18))
    → spatial query: rows 0-13 cached, rows 14-21 missing
    → compute ONLY rows 14-21 from decoder
    → assemble: rows 3-13 from cache + rows 14-21 from new computation
    → cached, rc=1
  ...

Each output row reuses ~60% of the previous upstream request.
Zero redundant decoding.
```

### 3.3 Three-Phase Execution Model

Pipeline execution has three distinct phases: a forward info pass, a backward
request pass, and a forward pixel pass. Understanding this flow is essential
for implementing nodes, plugins, and GPU/ML acceleration.

#### Phase 1: Forward Info Pass (graph build time)

When nodes are added to the graph, each computes its output `info()` — dimensions,
pixel format, color space. This flows **forward** through the graph:

```
Source (4000x3000 Rgb8)
  → Resize (2000x1500 Rgb8)    ← info() changes dimensions
  → Blur (2000x1500 Rgb8)      ← info() passes through (same dims)
  → Contrast (2000x1500 Rgb8)  ← info() passes through
  → Encoder                    ← sees final 2000x1500, determines tile geometry
```

Info is available **without computing any pixels**. The encoder/sink uses the
final info to plan its output tile geometry (JPEG: MCU rows, PNG: scanlines,
TIFF: native tiles).

#### Phase 2: Backward Request Pass (tile request time)

When the encoder requests a tile, the request propagates **backward** through
the graph. Each node maps the output rect to the input rect it needs via
`input_rect()`:

```
Encoder requests tile (0, 0, 2000, 16)
  ← Contrast: point-op, no expansion → requests (0, 0, 2000, 16)
  ← Blur: spatial op, needs overlap → requests (0, 0, 2000, 32)
     (expanded by blur radius, clamped to image bounds)
  ← Resize: coordinate transform → requests (0, 0, 4000, 64)
     (maps 2000x32 output coords back to 4000x64 source coords)
  ← Source: provides (0, 0, 4000, 64)
```

Each node type has its own `input_rect` mapping:
- **Point-ops** (brightness, contrast, gamma): identity — output rect = input rect
- **Spatial ops** (blur, sharpen, median): expand by kernel radius
- **Resize**: scale coordinates by inverse of resize ratio
- **Distortion** (barrel, swirl): compute bounding box of inverse transform
- **Crop**: offset by crop origin
- **Flip/rotate**: remap coordinates in the opposite direction

#### Phase 3: Forward Pixel Pass (computation)

Pixels flow **forward** from source to sink. Each node receives upstream
pixels for the region it requested and produces its output:

```
Source decodes (0, 0, 4000, 64) → 4000x64 pixels
  → Resize resamples to (0, 0, 2000, 32) → 2000x32 pixels
  → Blur processes with overlap, outputs (0, 0, 2000, 16) → 2000x16 pixels
     (crops the overlap it requested back to the encoder's tile)
  → Contrast applies LUT to (0, 0, 2000, 16) → 2000x16 pixels
  → Encoder receives the 2000x16 tile
```

The spatial/layer cache intercepts at each node boundary — if a node's output
for a given region is already cached, the backward/forward passes stop there.

#### Execution Strategy per Node Type

Nodes declare their execution strategy, which determines how they participate
in the three-phase flow:

| Strategy | info() | input_rect() | Pixel computation |
|----------|--------|-------------|-------------------|
| **Fusable point-op** | Pass-through | Identity (no expansion) | Engine applies fused LUT — node never computes directly |
| **Spatial (CPU)** | Pass-through | Expand by kernel radius | `compute_region` with upstream pixels |
| **Spatial (GPU)** | Pass-through | Full image (current) | GPU shader; layer cache serves tiles |
| **Resize/transform** | New dimensions | Inverse coordinate transform | Resampling from upstream |
| **Generator** | Declares output dims | N/A (no upstream) | Generates pixels from params |
| **ML** | Depends on model | Depends on model | Host ML runtime |

#### GPU Execution and Tiling

GPU ops currently expand `input_rect` to the full image. The GPU shader
processes the entire image in one dispatch. The layer cache stores the
full result, and subsequent tile requests from the encoder are served by
cropping from the cached output. This means:

- GPU runs **once** per execution (not per tile)
- Memory cost: full image cached (not tile-sized)
- Compute cost: optimal (single dispatch)
- Future optimization: tiled GPU execution would reduce memory by processing
  GPU-sized tiles with `input_rect` expansion, negotiating tile size against
  GPU texture limits and encoder geometry. Deferred — current approach works.

#### LUT Fusion and the Fused Node

When the graph contains consecutive fusable point-ops, the pipeline optimizer
collapses them into a single `FusedLutNode`:

```
Before fusion:
  Source → Brightness → Contrast → Gamma → Encoder
  (3 separate LUT applications, 3 pixel passes)

After fusion:
  Source → FusedLutNode(composed_lut) → Encoder
  (1 LUT application, 1 pixel pass)
```

The composed LUT is `lut_c[lut_b[lut_a[i]]]` — function composition via table
lookup. The `FusedLutNode` has both CPU (`apply_lut`) and GPU (LUT shader)
implementations. Individual point-op nodes are replaced entirely — they don't
execute.

---

### 3.3b Pipeline Composition and Analysis Sinks

Some operations need to inspect pixel content before determining parameters
for downstream nodes. For example, auto-crop needs to scan the image to find
edges before it knows the crop rect. This creates a chicken-and-egg problem:
`info()` runs at build time, but the output dimensions depend on pixels.

**Solution: analysis sinks.** An analysis sink is a sink node (like an encoder)
that consumes pixels and outputs structured parameters instead of encoded bytes.
The pipeline is composed in two stages — analysis first, then processing with
the analysis results.

#### Sink Node Types

The pipeline has two kinds of sink nodes:

| Sink type | Pixels in | Output |
|-----------|-----------|--------|
| **Encoder sink** | Full image (tiled) | Encoded bytes (JPEG, PNG, etc.) |
| **Analysis sink** | Full image (or sampled) | Structured params (crop rect, levels, etc.) |

Both drive execution the same way — they pull tiles backward through the
graph. The only difference is what they produce.

#### Pipeline Composition Pattern

Analysis sinks enable a two-stage pattern where the first stage's output
parameterizes the second stage:

```
// Stage 1: analysis — runs immediately, returns params
src = pipeline.read(data)
crop_rect = pipeline.analyze_auto_crop(src)     // analysis sink → {x,y,w,h}

// Stage 2: processing — uses params from stage 1
cropped = pipeline.crop(src, crop_rect)         // concrete params, info() is correct
resized = pipeline.resize(cropped, 800, 600)
output = pipeline.write_jpeg(resized, config)   // encoder sink → bytes
```

`analyze_auto_crop` triggers execution (like `write` does) — it pulls tiles
through the graph, inspects the pixels, and returns a result. Then `crop` is
built with known dimensions. No deferred info, no special barriers.

#### Generalizes Beyond Crop

The analysis sink pattern applies to any operation that needs content-aware
parameterization:

| Analysis sink | Output params | Feeds into |
|---------------|--------------|-----------|
| `analyze_auto_crop` | crop rect (x, y, w, h) | `crop` node |
| `analyze_auto_levels` | black/white/gamma | `levels` node |
| `analyze_white_balance` | temperature, tint | `white_balance` node |
| `analyze_histogram` | histogram data | `curves` or `equalize` node |
| `analyze_dominant_color` | color value(s) | `gradient_map` or palette |
| `analyze_exposure` | EV adjustment | `exposure` node |
| ML analysis (via MlOp) | face rects, masks, labels | selective adjustments, masks |

Each analysis sink is a thin wrapper: run the graph, collect statistics or
detections from the pixel data, return structured output. The processing
pipeline then uses those outputs as ordinary node configuration.

#### Why Not a Special Node?

An alternative design would be a "deferred info" node that materializes itself
mid-graph. This was rejected because:

1. It breaks the forward info pass — downstream nodes can't know their
   dimensions at build time
2. It requires special pipeline machinery (barriers, re-planning)
3. It conflates analysis (read pixels, produce params) with transformation
   (read pixels, produce pixels)

The analysis sink keeps the pipeline model simple: sinks consume pixels and
produce output. Encoders produce bytes. Analysis sinks produce params. Both
drive execution identically.

---

### 3.3c Composite Nodes (Pipeline-as-Node)

A composite node is a node whose implementation is itself a pipeline. It
contains an internal graph with special input/output reference nodes:

- **Input ref**: placeholder node that receives pixels from the outer pipeline
  (acts as the source within the internal graph)
- **Output ref**: the final node whose output becomes the composite's output
- **Internal graph**: arbitrary pipeline between input and output refs

```
Outer pipeline:  Source → [CompositeNode] → Resize → Encoder

CompositeNode internals:
  InputRef → NodeA → NodeB → ... → OutputRef
```

From the outer pipeline's perspective, a composite node is just a node. It
has `info()`, `input_rect()`, and `compute_region()`. These are derived from
its internal graph — `info()` returns the output ref's info, `input_rect()`
traces backward through the internal graph to determine what the input ref
needs.

#### Analysis + Processing Composites

Composite nodes naturally support the analysis sink pattern. The internal
graph can contain both analysis (materialize and inspect) and processing:

```
// auto_crop composite:
InputRef → analyze_edges(InputRef)        // analysis: scans pixels, determines crop rect
         → crop(InputRef, analyzed_rect)  // processing: crops using determined params
         → OutputRef
```

The composition is opaque to the outer pipeline. `auto_crop` is just a node
that takes an image and produces a cropped image.

#### Composites as Presets/Recipes

Presets and filter recipes are composite nodes with fixed params:

```
// "Vintage Film" preset composite:
InputRef → curves(vintage_curve_params)
         → color_balance(warm_shift)
         → film_grain(amount=0.3, size=1.5)
         → vignette(strength=0.4)
         → OutputRef
```

A preset is a serialized composite node definition — a pipeline fragment with
input/output refs and predetermined configuration. Loading a preset creates
a composite node. This unifies presets, recipes, macros, and multi-step
operations under one mechanism.

#### Nested Composition

Composite nodes can contain other composite nodes. The three-phase execution
model works recursively:

```
// "Portrait Enhance" composite contains the "Auto Levels" composite:
InputRef → [AutoLevels composite] → [SkinSmooth composite] → Sharpen → OutputRef
```

Each level resolves its own info/input_rect/compute_region through its
internal graph. The outer pipeline doesn't know or care about nesting depth.

#### Plugin Composites

Plugins (external .wasm components) can define composite nodes. The plugin
exports a composite definition (internal graph description), and the host
constructs the graph. This allows plugins to provide multi-step operations
without the host knowing the implementation details.

---

### 3.3d Node Trait (Revised)

```rust
/// The core trait every pipeline node implements.
trait ImageNode {
    /// Image dimensions and format (available without computing pixels).
    fn info(&self) -> &ImageInfo;

    /// Compute a tile into the provided buffer. The node acquires upstream
    /// tiles from the pool as needed and releases them when done.
    fn compute_tile(
        &self,
        request: &TileRequest,
        output: &mut Vec<u8>,
        pool: &mut TilePool,
        graph: &NodeGraph,
    ) -> Result<(), ImageError>;

    /// Overlap this node needs from upstream for each output tile.
    fn overlap(&self) -> Overlap;

    /// Access pattern hint — helps the pool optimize caching.
    fn access_pattern(&self) -> AccessPattern;
}

enum AccessPattern {
    /// Output tiles map to same-position upstream tiles (point ops, crop).
    Sequential,
    /// Output tiles map to shifted/scaled upstream tiles (resize, blur).
    LocalNeighborhood,
    /// Output tiles may request any upstream region (rotation, flip).
    RandomAccess,
    /// Must see all upstream tiles before producing any output (histogram eq).
    GlobalTwoPass,
    /// Has multiple upstream nodes (composite, blend).
    MultiInput,
}
```

### 3.4 Tile Request Propagation

When `write()` is called, it iterates output tiles and each request cascades:

```
write_jpeg(node_id=2, config)
  for each output strip (y=0, y=64, y=128, ...):
    handle = pool.acquire(TileKey{node:2, y, h:64}, |buf| {
      // blur (node 2) computes by acquiring from upstream
      src_h0 = pool.acquire(TileKey{node:0, y:y-overlap, h:64+2*overlap}, |buf| {
        // source decoder (node 0) decodes the region
        decoder.decode_region(y-overlap, 64+2*overlap, buf);
      });
      blur_kernel(pool.read(src_h0), buf);
      pool.release(src_h0);
    });
    encoder.consume_strip(pool.read(handle));
    pool.release(handle);
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

### 3.5 Operation Validation Against Tile Pool

Every operation from the parity matrix validated against the tile pool + ref-count
model. Each entry documents access pattern, pool behavior, and any caveats.

#### Tier 1: Works perfectly (sequential access, minimal pool)

| Operation | Access Pattern | Pool Slots Needed | Notes |
|-----------|---------------|-------------------|-------|
| Brightness | Sequential | 1 upstream + 1 output | Point op, zero overlap |
| Contrast | Sequential | 1 + 1 | Point op |
| Gamma | Sequential | 1 + 1 | Point op |
| Invert/negate | Sequential | 1 + 1 | Point op |
| Grayscale | Sequential | 1 + 1 | Per-pixel channel combine |
| Threshold | Sequential | 1 + 1 | Point op |
| Levels/curves | Sequential | 1 + 1 | LUT-based point op |
| Color space convert | Sequential | 1 + 1 | Per-pixel transform |
| Crop | Sequential | 1 + 1 | Coordinate offset only |
| Embed/pad | Sequential | 0-1 + 1 | Tiles outside input = fill color, no upstream |
| Flip horizontal | Sequential | 1 + 1 | Mirror column coordinates within same strip |
| Rotate 180 | Sequential (reverse) | 1 + 1 | Reverse strip order + reverse columns |

#### Tier 2: Works with sliding window (local neighborhood, 2-4 pool slots)

| Operation | Access Pattern | Pool Slots Needed | Overlap |
|-----------|---------------|-------------------|---------|
| Blur (gaussian) | LocalNeighborhood | 2-3 upstream + 1 output | ceil(3*sigma) per side |
| Sharpen (unsharp mask) | LocalNeighborhood | 2-3 + 1 | 1-3px |
| Median filter | LocalNeighborhood | 2-3 + 1 | radius |
| Custom convolution | LocalNeighborhood | 2-3 + 1 | kernel_size/2 |
| Sobel/Prewitt edge detect | LocalNeighborhood | 2-3 + 1 | 1px (3x3 kernel) |
| Morphology (erode/dilate) | LocalNeighborhood | 2-3 + 1 | structuring element radius |
| Emboss | LocalNeighborhood | 2-3 + 1 | 1px |
| Resize (any filter) | LocalNeighborhood | 2-3 + 1 | ceil(filter_radius * scale_factor) |

The sliding window works because strips are processed top-to-bottom. The previous
upstream strip stays cached (rc > 0) while the next strip is computed, providing
the overlap region. Only 2-3 upstream strips are ever live simultaneously.

#### Tier 3: Works with expanded pool (random access)

| Operation | Issue | Pool Behavior | Memory Impact |
|-----------|-------|---------------|---------------|
| Rotate 90/270 | Output strip N needs a vertical slice across ALL source rows | Source fully decoded into pool (or seek cache provides rows on demand) | Up to full image for sequential source formats |
| Flip vertical | First output strip needs LAST source strip | Source fully decoded in reverse order | Same as rotation |
| Arbitrary angle rotation | Any output pixel maps to any source pixel | Source fully cached | Full image |
| Affine transform | General coordinate remapping | Source fully cached | Full image |
| Perspective/distort | Arbitrary mapping | Source fully cached | Full image |

**Not a design flaw — inherent to the operation.** These operations fundamentally
require random access to the full source. The pool handles it by caching all source
strips. Memory usage equals the full image for these operations, same as the current
architecture, but downstream operations still benefit from tiling.

**Mitigation:** For `rotate90(read(large_png)) → write()`, the source decoder
fills pool slots on demand via the seek cache. The pool may hold the full decoded
image, but it does so in reusable slots. Once the rotate is complete and output
is written, all slots are released.

#### Tier 4: Works with two-pass (global operations)

| Operation | First Pass | Second Pass | Notes |
|-----------|-----------|-------------|-------|
| Histogram equalize | Iterate ALL upstream tiles, compute histogram | Apply LUT per tile | Stats pass acquires+releases each tile sequentially |
| Auto-levels/auto-contrast | Compute min/max/mean across all tiles | Apply linear mapping | Same pattern |
| Smart crop | Compute saliency/interest map across all tiles | Crop to detected region | Detection pass + crop pass |
| Trim/autocrop | Scan border tiles for uniform color | Crop to content bounds | Partial scan (edges only) |

**How it works in the pool:**
1. First pass: acquire tile (rc=1), read pixels, compute stats, release tile (rc=0→reclaimable)
2. Between passes: only the computed statistics are held (tiny: histogram = 256 entries)
3. Second pass: acquire tiles again (recomputed or cache-hit if still in pool), apply transform

**Key: the two-pass pattern does NOT hold all tiles simultaneously.** Each tile is
acquired and released during the stats pass. The pool size stays small. Only the
statistics summary persists between passes.

#### Tier 5: Works with multiple inputs (DAG nodes)

| Operation | Inputs | Pool Behavior | Notes |
|-----------|--------|---------------|-------|
| Alpha composite | 2 source nodes | Acquire tile from node A + tile from node B, composite, release both | rc=1 each |
| Porter-Duff blend | 2 sources | Same as composite | All blend modes |
| Watermark overlay | 2 sources | Overlay image tile (often smaller, so many output tiles need no overlay tile) | Sparse upstream |
| Image concatenation | N sources | Output tile maps to whichever source covers that region | Only 1 source tile per output tile |
| Difference/comparison | 2 sources | Acquire both, compute diff, release | |

The pool handles DAG topology naturally. Multiple `acquire()` calls for different
node_ids in the same output tile computation. Each upstream tile has its own
TileKey with its node_id, so caching is independent.

#### Tier 6: Encoder-specific considerations

| Encoder | Tile Compatibility | Notes |
|---------|-------------------|-------|
| JPEG | Perfect | MCUs are 8x8 blocks; 64-row strips = 8 MCU rows, perfect alignment |
| PNG | Perfect | Scanline-by-scanline encoding; strips provide scanlines in order |
| TIFF (tiled) | Perfect | Write tiles directly from pool tiles |
| TIFF (stripped) | Perfect | Write strips directly |
| WebP (lossy) | Caveat | VP8 encoder typically needs full image; may need to materialize |
| WebP (lossless) | Caveat | VP8L may need full image |
| AVIF (tiled) | Perfect | AV1 tiles can be encoded independently |
| AVIF (single tile) | Caveat | Needs full image |
| GIF | Caveat | LZW encoder can be fed scanlines, but frame disposal may need prior frame |

**Encoder materialization strategy:** For encoders that need the full image (WebP,
single-tile AVIF), the `write()` sink iterates all upstream tiles into a contiguous
buffer before calling the encoder. This is a worst-case fallback, not a design flaw.
The pool still provides the tiles efficiently; only the final assembly step needs
the full image. As encoders improve or we add streaming encode support, this
fallback can be removed per-format.

#### Summary: No design-breaking issues found

| Access Pattern | Pool Behavior | Example Operations |
|----------------|---------------|--------------------|
| Sequential (point ops) | 2 slots, sliding | brightness, contrast, grayscale |
| Local neighborhood | 3-4 slots, sliding window | blur, sharpen, resize, edge detect |
| Random access | Many slots (up to full image) | rotate 90/270, flip vertical, affine |
| Global two-pass | 2 slots during stats, 2 during apply | histogram eq, auto-levels |
| Multi-input DAG | 2+ slots per source | composite, blend, watermark |
| Encoder (streaming) | Consumes tiles in order | JPEG, PNG, TIFF |
| Encoder (materialized) | Collects all tiles first | WebP lossy, single-tile AVIF |

**The tile pool design works for ALL operations.** Operations that fundamentally
need full-image access (rotation, global stats) use more pool slots, but the
design doesn't break — it gracefully degrades to current-architecture memory
usage for those cases while providing 50-100x improvement for the common case
(sequential transforms and filters).

### 3.6 Operation Fusion

The pipeline can detect optimization opportunities before executing:

| Pattern | Fusion | Savings |
|---------|--------|---------|
| read(JPEG) + resize(1/N) | JPEG DCT shrink-on-load | Decode at 1/2, 1/4, or 1/8 size |
| read(TIFF) + crop | Decode only the tiled region | Skip unused tiles |
| brightness + contrast | Single linear transform: `a * pixel + b` | One pass instead of two |
| multiple point ops | Compose into single LUT | One lookup per pixel |
| resize + resize | Compose into single resize | One resampling pass |

**Fusion detection:** After the node chain is built (before first `get_tile`), a fusion pass walks the chain looking for known patterns. Fused nodes replace the original chain segment.

### 3.5 Output Chunk Strategy (Write Sink Drives Geometry)

There is no fixed tile size. The **write sink** determines output chunk geometry
based on the target format, and each upstream node expands the request by its
kernel overlap. The geometry flows backward through the graph dynamically.

| Write Format | Output Chunk | Rationale |
|-------------|-------------|-----------|
| JPEG | Full-width, 8 or 16 rows (MCU-aligned) | JPEG encodes in MCU blocks |
| PNG | Full-width, 1 row (scanline) or batch | PNG filters per scanline |
| TIFF (tiled) | Native tile size (e.g., 256x256) | Direct tile write |
| TIFF (stripped) | Full-width, N rows per strip | Match strip config |
| WebP | Full image (materialized) | VP8 encoder needs full image |
| AVIF (tiled) | AV1 tile grid | Per-tile encoding |

The spatial cache ensures that upstream overlap regions computed for one output
chunk are automatically reused by the next chunk — no redundant computation
regardless of chunk geometry.

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

### D1: Pipeline resource owns the node graph; nodes are opaque IDs (not WIT resources)
WIT `own<>` transfers ownership (node consumed after one use) and `borrow<>` is
call-scoped only. Neither works for nodes that must persist across multiple tile
requests (blur on tile (0,0) and tile (1,0) both need the same upstream).
Solution: the pipeline resource holds `Vec<Box<dyn ImageNode>>` internally.
Transform functions accept and return `node-id` (u32). Nodes persist until the
pipeline is dropped.

### D2: Transform functions are pipeline methods, not node methods
`pipeline.resize(source_id, w, h, filter)` returns a new `node-id`. This keeps
all nodes in the pipeline's ownership and allows the pipeline to see the full
chain before execution (fusion).

### D2b: Tile identity is (node_id, x, y, w, h)
Tile at (0,0) from the source decoder (node 0) is a different pool entry than
tile at (0,0) from the blur node (node 2). Each node's output tiles are
independently cached and ref-counted in the pool.

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

## 8. Open Questions (Resolved) and Remaining

### RESOLVED: Q1 — Node ownership model
**Decision: Pipeline owns all nodes; node-id (u32) for references.**
WIT resource ownership/borrowing doesn't work for nodes that must persist across
multiple tile requests. See D1 above.

### RESOLVED: Q2 — Tile identity
**Decision: TileKey = (node_id, x, y, w, h).**
Tiles are scoped to the producing node. The same (x,y) coordinates at different
pipeline steps are different cache entries. See D2b above.

### RESOLVED: Q3 — How to handle global operations
**Decision: Two-pass node with stats caching.**
First pass iterates all upstream tiles (acquire/release each) to compute statistics.
Second pass applies the transform. Pool size stays small — tiles aren't held
simultaneously during the stats pass. See Tier 4 in operation validation.

### RESOLVED: Q4 — Lazy vs immediate
**Decision: Lazy with explicit write() trigger.**
Node chain is built without computation. First `write()` call triggers execution.
This enables fusion detection before any pixels are computed.

### RESOLVED: Q5 — Parallelism strategy (dual-level API + portable orchestrator)

**Decision: Dual-level API with deployment-portable orchestrator.**

The component exposes two API levels:

**Level 1 — Pipeline resource (single-threaded, works today):**
The component owns graph, cache, execution. Simple. Works everywhere.
```wit
resource pipeline {
    read: func(data: buffer) -> result<node-id, rasmcore-error>;
    resize: func(source: node-id, ...) -> node-id;
    write-jpeg: func(source: node-id, ...) -> result<buffer, rasmcore-error>;
}
```

**Level 2 — Stateless compute kernels (host-parallelizable):**
Pure functions: pixels in, pixels out. No graph, no cache, no state.
```wit
interface compute {
    apply-blur: func(pixels: buffer, info: image-info, radius: f32) -> result<buffer, rasmcore-error>;
    apply-resize: func(pixels: buffer, info: image-info, w: u32, h: u32, filter: resize-filter) -> result<buffer, rasmcore-error>;
}
```

**Parallelism phases:**

| Phase | Mechanism | Where | Prerequisite |
|-------|-----------|-------|-------------|
| 1 | Single-threaded pipeline | Inside WASM component | None (current) |
| 2 | Host-driven parallelism | Host manages graph+cache+thread pool, dispatches to N WASM instances via Level 2 compute kernels | None — works today |
| 3 | Internal parallelism | Same orchestrator code compiled to WASM, uses wasi-threads internally | wasi-threads stabilization |
| 4 | SIMD | WASM SIMD intrinsics in compute kernels | Available now |

**Key principle: the orchestrator (graph + spatial cache + dispatch) is pure
Rust that compiles to both native and WASM.** In Phase 2, it runs on the host.
In Phase 3, the exact same code compiles into the WASM component via
`cargo component build`. Zero code changes — just a different compile target.

**Host-driven parallelism (Phase 2) flow:**
1. Host builds node graph (mirrors pipeline structure)
2. Host manages spatial cache in native memory (RwLock for thread safety)
3. Host dispatches tile computations to thread pool
4. Each thread calls a WASM instance's Level 2 compute kernel (pixels in → pixels out)
5. Host stores results in shared cache
6. No WASM shared memory needed — the host IS the shared memory layer

**Why this is better than wasi-threads alone:**
- Works TODAY without waiting for wasi-threads
- Host-side cache is in native memory (no WASM linear memory constraints)
- WASM instances are stateless — no shared mutable state, no data races by construction
- When wasi-threads arrives, the same architecture moves inside the component seamlessly

### REMAINING: Q6 — Pool size heuristics
How should the pool auto-size? Options:
- Fixed pool (e.g., 32 slots) — simple, predictable memory ceiling
- Adaptive based on access pattern hints — grows for random access ops, shrinks for sequential
- User-configurable memory budget — pool calculates max slots from budget + tile size
**Recommendation: Start with fixed pool + user-configurable max. Add adaptive later.**

### REMAINING: Q7 — Canny edge detection hysteresis
Canny's hysteresis thresholding uses connected component analysis which is
inherently global — connectivity can span the entire image. Options:
- Generous overlap (32px) with approximate tile-boundary handling
- Full-image materialization for exact results
- Two-phase: tile-local hysteresis + boundary stitching pass
**Recommendation: Start with materialization; optimize with overlap later if needed.**

### REMAINING: Q8 — FFT / frequency domain
Full-image FFT requires all pixels. Options:
- Overlap-save method for FFT-based convolution (block-wise, compatible with tiles)
- Full materialization for direct frequency access (viewing power spectrum)
**Recommendation: Overlap-save for convolution use cases. Full materialization for
direct FFT access. Both work within the tile pool model.**
