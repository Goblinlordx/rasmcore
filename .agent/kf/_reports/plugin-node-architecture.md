# Plugin Node Architecture — External .wasm Components as Pipeline Nodes

> Design document for the rasmcore plugin system.
> Updated: 2026-04-02

## 1. Overview

Plugins are external `.wasm` components that implement the same interface as
built-in pipeline nodes. The host loads them at runtime using its own WASM
runtime (wasmtime for CLI, browser native WASM for web). Zero size overhead
in the core component.

A plugin is a full pipeline node. It can be a filter, transform, encoder,
decoder, or analysis sink. The pipeline graph treats plugin nodes identically
to built-in nodes — the only difference is where the implementation comes
from.

### Key Design Principles

1. **Same interface** — plugins export the same contract as built-in nodes
2. **Host loads, host executes** — the core component never loads plugins
3. **Execution strategies are mutually exclusive** — fusable / GPU / ML / CPU-only
4. **Composites can reference built-in nodes** — plugins define compositions, not reimplementations
5. **Remote bundles on demand** — plugins fetched from URLs, cached locally

---

## 2. Current Node Interface (What Plugins Must Implement)

### 2.1 ImageNode Trait

The core contract every pipeline node implements:

```rust
pub trait ImageNode {
    fn info(&self) -> ImageInfo;
    fn compute_region(&self, request: Rect, upstream_fn: ...) -> Result<Vec<u8>, ImageError>;
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect;
    fn access_pattern(&self) -> AccessPattern;

    // Optional capabilities:
    fn as_point_op_lut(&self) -> Option<[u8; 256]>;      // fusable point-op
    fn as_color_lut_op(&self) -> Option<ColorLut3D>;      // fusable color op
    fn as_affine_op(&self) -> Option<([f64; 6], u32, u32)>;  // affine fusion
}
```

### 2.2 Capability Traits (Optional)

```rust
pub trait GpuCapable {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>>;
}

pub trait InputRectProvider {
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect;
}
```

### 2.3 Config Params

Each filter has a typed config struct with param metadata:
- Field name, type (f32, u8, bool, enum)
- Min/max/step/default values
- UI hints

---

## 3. Plugin Execution Strategies

A plugin declares ONE execution strategy. These are mutually exclusive:

### 3.1 Fusable Point-Op

- Plugin exports: `build_point_lut(config) -> [u8; 256]`
- `compute_region` NOT required
- Engine fuses with adjacent point-ops into single LUT
- Fused LUT runs on GPU (engine provides LUT shader) or CPU automatically
- Use for: brightness, contrast, gamma, curves, any per-channel mapping

### 3.2 GPU-Accelerated (Non-Fusable)

- Plugin exports: `compute_region` + `gpu_ops(w, h) -> Option<Vec<GpuOp>>`
- `compute_region` is CPU fallback
- Plugin provides WGSL shader source + params
- Host dispatches to GPU; falls back to CPU if unavailable
- `input_rect` required for spatial ops
- Use for: blur, bilateral, custom spatial filters

### 3.3 ML-Accelerated

- Plugin exports: `ml_op(w, h) -> Option<MlOp>`
- No CPU fallback — ML features are ML-only
- Plugin provides model reference + tensor params
- Host dispatches to ONNX Runtime / CoreML / WebNN
- Use for: background removal, super resolution, style transfer

### 3.4 CPU-Only

- Plugin exports: `compute_region` + `input_rect`
- No GPU or ML acceleration
- Default strategy if no other capabilities declared
- Use for: simple operations, prototyping

---

## 4. Plugin WIT Interface

### 4.1 What a Plugin Exports

```wit
interface rasmcore-plugin {
    // ─── Required ────────────────────────────────────────
    
    /// Plugin manifest — lists all filters/nodes provided
    get-manifest: func() -> plugin-manifest;
    
    // ─── Per-Filter Exports (by filter name) ─────────────
    
    /// Get config schema for a filter
    get-config-schema: func(filter-name: string) -> config-schema;
    
    /// Compute pixels for a region (CPU path)
    /// Not required for fusable point-ops
    compute-region: func(
        filter-name: string,
        config: list<config-entry>,
        request: rect,
        upstream-pixels: list<u8>,
        upstream-width: u32,
        upstream-height: u32,
        info: image-info,
    ) -> result<list<u8>, string>;
    
    /// Input rect expansion (spatial ops)
    input-rect: func(
        filter-name: string,
        config: list<config-entry>,
        output: rect,
        bounds-w: u32,
        bounds-h: u32,
    ) -> rect;
    
    // ─── Optional Capability Exports ─────────────────────
    
    /// Build 1D LUT for fusable point-op
    build-point-lut: func(
        filter-name: string,
        config: list<config-entry>,
    ) -> option<list<u8>>;  // 256 entries or None
    
    /// GPU shader for accelerated execution
    gpu-ops: func(
        filter-name: string,
        config: list<config-entry>,
        width: u32,
        height: u32,
    ) -> option<list<gpu-op>>;
    
    /// ML operation for inference dispatch
    ml-op: func(
        filter-name: string,
        config: list<config-entry>,
        width: u32,
        height: u32,
    ) -> option<ml-op>;
}
```

### 4.2 What a Plugin Imports (Host-Provided)

Plugins don't need imports for basic operation. The host provides upstream
pixels as arguments to `compute-region`. For composite plugins that reference
built-in nodes, the host provides:

```wit
interface rasmcore-host {
    /// Execute a built-in node by name (for composite plugins)
    execute-builtin: func(
        filter-name: string,
        config: list<config-entry>,
        pixels: list<u8>,
        info: image-info,
    ) -> result<list<u8>, string>;
}
```

### 4.3 Type Definitions

```wit
record plugin-manifest {
    name: string,
    version: string,
    author: option<string>,
    filters: list<filter-info>,
}

record filter-info {
    name: string,
    category: string,
    description: string,
    strategy: execution-strategy,
    access-pattern: access-pattern-hint,
}

enum execution-strategy {
    fusable-point-op,
    gpu-accelerated,
    ml-accelerated,
    cpu-only,
}

enum access-pattern-hint {
    sequential,
    local-neighborhood,
    random-access,
    global-two-pass,
}

record config-schema {
    fields: list<config-field>,
}

record config-field {
    name: string,
    field-type: param-type,
    min: option<f64>,
    max: option<f64>,
    step: option<f64>,
    default-value: option<f64>,
    hint: option<string>,
}

record config-entry {
    name: string,
    value: config-value,
}

variant config-value {
    float-val(f64),
    int-val(s64),
    bool-val(bool),
    string-val(string),
}
```

---

## 5. Plugin Bundle Format

### 5.1 Bundle Structure

A plugin bundle is a directory or archive containing:

```
my-plugin/
  manifest.json       # Plugin metadata + filter list
  plugin.wasm         # Compiled WASM component
  README.md           # Optional documentation
```

### 5.2 Manifest Schema

```json
{
  "name": "vintage-film-pack",
  "version": "1.0.0",
  "author": "Example Author",
  "min_engine_version": "0.2.0",
  "sha256": "abc123...",
  "size": 245760,
  "filters": [
    {
      "name": "film_fade",
      "category": "grading",
      "strategy": "fusable-point-op",
      "description": "Film fade effect with configurable intensity"
    },
    {
      "name": "vintage_blur",
      "category": "spatial",
      "strategy": "gpu-accelerated",
      "description": "Soft vintage blur with chromatic fringing"
    }
  ]
}
```

### 5.3 Distribution

- Plugins hosted at configurable registry URL(s)
- Host downloads on first use, caches in platform-appropriate location:
  - CLI: `~/.rasmcore/plugins/`
  - Browser: IndexedDB or Cache API
- Integrity verified via sha256 hash in manifest
- Version checking: `min_engine_version` prevents loading incompatible plugins

---

## 6. Plugin Loading

### 6.1 CLI Host (wasmtime)

```rust
// 1. Discover plugins (scan directory or fetch manifest)
let plugins = discover_plugins(&plugin_dir);

// 2. Load and instantiate each plugin component
for plugin_path in plugins {
    let component = Component::from_file(&engine, &plugin_path)?;
    let instance = linker.instantiate(&mut store, &component)?;
    
    // 3. Read manifest and register filters
    let manifest = instance.call_get_manifest(&mut store)?;
    for filter in manifest.filters {
        registry.register_plugin_filter(filter, instance.clone());
    }
}

// 4. Plugin filters now appear in list_filters() and param-manifest
```

### 6.2 Browser Host

```javascript
// 1. Fetch plugin WASM
const response = await fetch(pluginUrl);
const bytes = await response.arrayBuffer();

// 2. Instantiate (via jco component polyfill or native Component Model)
const instance = await WebAssembly.instantiate(bytes, imports);

// 3. Read manifest and register
const manifest = instance.exports.getManifest();
for (const filter of manifest.filters) {
    registry.registerPluginFilter(filter, instance);
}
```

### 6.3 Discovery Protocol

1. **Local directory**: scan `plugins/` directory for `manifest.json` files
2. **Registry URL**: fetch `{registry_url}/index.json` for available plugins
3. **Direct URL**: load a specific plugin by URL (for development/testing)

---

## 7. Pipeline Integration

### 7.1 Plugin Node Wrapper

The host wraps each plugin filter in an `ImageNode` implementation:

```rust
struct PluginNode {
    plugin_instance: Arc<PluginInstance>,
    filter_name: String,
    config: Vec<ConfigEntry>,
    upstream: u32,
    source_info: ImageInfo,
    strategy: ExecutionStrategy,
}

impl ImageNode for PluginNode {
    fn info(&self) -> ImageInfo { self.source_info.clone() }
    
    fn compute_region(&self, request: Rect, upstream_fn: ...) -> Result<Vec<u8>> {
        // Fetch upstream pixels
        let upstream_pixels = upstream_fn(self.upstream, self.expanded_request(request))?;
        // Call into plugin WASM component
        self.plugin_instance.call_compute_region(
            &self.filter_name, &self.config, request,
            &upstream_pixels, expanded.width, expanded.height, &self.source_info,
        )
    }
    
    fn input_rect(&self, output: Rect, bw: u32, bh: u32) -> Rect {
        self.plugin_instance.call_input_rect(
            &self.filter_name, &self.config, output, bw, bh,
        )
    }
    
    fn as_point_op_lut(&self) -> Option<[u8; 256]> {
        if self.strategy != ExecutionStrategy::FusablePointOp { return None; }
        self.plugin_instance.call_build_point_lut(&self.filter_name, &self.config)
            .map(|v| {
                let mut lut = [0u8; 256];
                lut.copy_from_slice(&v);
                lut
            })
    }
    
    fn access_pattern(&self) -> AccessPattern { ... }
}

impl GpuCapable for PluginNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if self.strategy != ExecutionStrategy::GpuAccelerated { return None; }
        self.plugin_instance.call_gpu_ops(&self.filter_name, &self.config, width, height)
    }
}
```

### 7.2 LUT Fusion Integration

Plugin fusable point-ops participate in the same fusion chain as built-in
point-ops:

```
Built-in Brightness → Plugin "film_fade" → Built-in Contrast
                    ↓ (all fusable)
              FusedLutNode(composed_lut)
              → GPU LUT shader or CPU apply_lut
```

The pipeline optimizer calls `as_point_op_lut()` on each node. Plugin nodes
return their LUT from the WASM call. Fusion composes all LUTs into one. The
plugin's `compute_region` is never called.

### 7.3 GPU Dispatch Integration

Plugin GPU ops flow through the same dispatch path as built-in GPU ops:

1. Graph walker checks `GpuCapable::gpu_ops()` on each node
2. Plugin node calls into WASM to get shader source + params
3. Shader is dispatched to GPU alongside built-in GPU ops
4. Can be batched with adjacent built-in GPU ops (same ping-pong buffer chain)

### 7.4 Composite Plugins

A plugin can define composite nodes — internal pipeline graphs that reference
both plugin-defined and built-in nodes.

```json
{
  "name": "vintage_film_look",
  "type": "composite",
  "graph": [
    { "id": 0, "type": "input-ref" },
    { "id": 1, "type": "builtin", "name": "curves_master",
      "config": { "points": [[0,10],[64,50],[192,220],[255,245]] },
      "upstream": 0 },
    { "id": 2, "type": "builtin", "name": "color_balance",
      "config": { "shadow_r": 10, "shadow_g": -5, "shadow_b": 15 },
      "upstream": 1 },
    { "id": 3, "type": "builtin", "name": "film_grain",
      "config": { "amount": 0.3, "size": 1.5 },
      "upstream": 2 },
    { "id": 4, "type": "builtin", "name": "vignette",
      "config": { "strength": 0.4 },
      "upstream": 3 },
    { "id": 5, "type": "output-ref", "upstream": 4 }
  ]
}
```

The host resolves `"builtin"` node types against its registry. The composite
executes as a sub-graph. The plugin ships NO code for these operations —
it only defines the composition.

Composites can mix built-in and plugin-defined nodes:

```json
{ "id": 2, "type": "plugin", "name": "my_custom_effect",
  "config": { "intensity": 0.7 }, "upstream": 1 }
```

---

## 8. Performance Analysis

### 8.1 Cross-Component Call Overhead

The WASM Component Model enforces shared-nothing isolation. Each
`compute_region` call copies the pixel buffer across component boundaries.

| Tile size | Buffer size | Copy overhead |
|-----------|------------|---------------|
| 256x16 (JPEG MCU rows) | 16 KB | ~2 us |
| 512x512 | 1 MB | ~100-500 us |
| 4000x16 (full-width strip) | 256 KB | ~25-100 us |
| 4000x3000 (full image) | 48 MB | ~5-15 ms |

### 8.2 Mitigation Strategies

1. **Tile pipeline limits copy size** — buffers are tile-sized, not full-image.
   A typical JPEG MCU row (4000x16 RGBA) is 256KB — copy overhead is <100us,
   negligible vs. actual filter computation.

2. **LUT fusion eliminates copies entirely** — fusable plugin point-ops are
   collapsed into a composed LUT. The plugin's `build_point_lut` is called
   once (256 bytes), then the fused LUT runs on GPU/CPU. No per-tile
   cross-component call.

3. **GPU shader dispatch avoids pixel copies** — GPU-capable plugin nodes
   provide shader source (a string), not pixel data. The shader runs on GPU
   hardware. Only the small params buffer crosses the component boundary.

4. **Composites using built-in nodes have zero crossing** — composite plugins
   that reference only built-in nodes execute entirely within the core
   component. The plugin only provides the graph definition (once, at load
   time).

### 8.3 When Overhead Matters

The only case where per-tile copy overhead is significant is a **CPU-only
plugin spatial filter** processing large tiles. For this case, the per-tile
overhead of ~100us on a 256KB buffer is acceptable given that the filter
computation itself typically takes 1-10ms.

Conclusion: cross-component overhead is not a practical concern for the
tile pipeline architecture.

---

## 9. Security Model

### 9.1 WASM Sandboxing

Plugins run in WASM sandboxes with:
- **No filesystem access** — plugins cannot read/write files
- **No network access** — plugins cannot make HTTP requests
- **No system calls** — plugins cannot spawn processes
- **Memory isolation** — each plugin has its own linear memory
- **Capability-based imports** — plugins can only call functions the host provides

### 9.2 Resource Limits

- **Memory budget**: host caps each plugin's linear memory (e.g., 256MB)
- **Execution timeout**: host applies wall-clock timeout per compute_region call
- **Fuel metering**: wasmtime fuel limits prevent infinite loops

### 9.3 Trust Levels

| Source | Trust | Verification |
|--------|-------|-------------|
| Built-in (compiled) | Full | Part of build |
| Official plugins (signed) | High | SHA256 + signature |
| Community plugins (unsigned) | Low | SHA256 only |
| User-provided (local) | User's discretion | None |

---

## 10. Plugin SDK (Future)

A plugin SDK would provide:

1. **Template project** — Cargo template with WIT bindings and example filter
2. **`rasmcore-plugin` crate** — Rust helper with macros for declaring filters:
   ```rust
   #[rasmcore_plugin::filter(
       name = "my_blur",
       category = "spatial",
       strategy = "gpu-accelerated",
   )]
   struct MyBlurParams {
       #[param(min = 0.1, max = 50.0, default = 3.0)]
       radius: f32,
   }
   
   impl MyBlurParams {
       fn compute_region(&self, request: Rect, upstream: &[u8], info: &ImageInfo) -> Vec<u8> { ... }
       fn input_rect(&self, output: Rect, bw: u32, bh: u32) -> Rect { ... }
       fn gpu_ops(&self, w: u32, h: u32) -> Option<Vec<GpuOp>> { ... }
   }
   ```
3. **Build tooling** — `cargo rasmcore-plugin build` compiles to `.wasm` component + manifest
4. **Test harness** — run plugin filters against reference images locally

---

## 11. Comparison: Built-in vs Plugin

| Aspect | Built-in | Plugin |
|--------|----------|--------|
| Distribution | Compiled into core .wasm | Separate .wasm file |
| Loading | Compile-time (inventory) | Runtime (host loader) |
| Performance | Native — no copy overhead | Copy overhead per tile (~100us) |
| LUT fusion | Direct LUT composition | Same (via build_point_lut call) |
| GPU accel | Direct shader construction | Same (via gpu_ops call) |
| Security | Full trust | WASM sandbox |
| Size impact | Increases core .wasm size | Zero impact on core |
| Update cycle | Rebuild core | Replace plugin .wasm |
| Development | Requires rasmcore build | Independent build |

---

## 12. Implementation Roadmap

### Phase 1: Core Plugin Loading (MVP)

1. Define plugin WIT interface
2. Implement plugin loader for CLI (wasmtime)
3. Plugin discovery from directory
4. PluginNode wrapper implementing ImageNode
5. Plugin filters appear in filter list
6. One example plugin (simple point-op)

### Phase 2: Capability Integration

1. Fusable point-op plugins (LUT fusion)
2. GPU-capable plugins (shader dispatch)
3. Plugin config schema and validation
4. Browser plugin loading (jco / native Component Model)

### Phase 3: Composites and Distribution

1. Composite plugin definitions (JSON graph)
2. Built-in node resolution in composites
3. Remote plugin bundle format
4. Download/cache/verify protocol
5. Plugin SDK template project

### Phase 4: ML Integration

1. ML-capable plugins (model-ref dispatch)
2. Model bundle loading (separate ML execution track)
3. Analysis sink plugins

---

## 13. Open Questions

1. **Plugin versioning**: how to handle breaking changes in the plugin WIT
   interface? Semantic versioning on the interface, with the host supporting
   N-1 versions?

2. **Plugin dependencies**: can a plugin depend on another plugin? For V1,
   plugins are independent. Plugin-to-plugin dependencies add significant
   complexity (load ordering, circular deps).

3. **Hot reload**: should the host support hot-reloading plugins without
   restarting? Useful for development, complex for production (in-flight
   pipeline state).

4. **Shared-everything linking**: when the Component Model spec adds shared
   memory between same-trust-domain components, plugins from the same
   publisher could share memory for zero-copy. Not available yet.
