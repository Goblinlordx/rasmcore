# Design: Pipeline Demand Regulator

Adaptive flow control for tiled pipeline execution.

## 1. Current State

### Demand Flow

```
Sink (write_tiled)
  |  iterates 512x512 grid
  v
request_region(node_id, tile_rect)
  |
  +-- Layer cache hit? --> return cached
  |
  +-- GPU available? --> input_rect(tile) --> request upstream --> GPU execute --> crop
  |
  +-- ML available?  --> request FULL image --> ML execute
  |
  +-- CPU fallback   --> node.compute_region(tile, upstream_fn) --> recursive pull
```

### Where Tile Size Is Decided

| Location | Value | Overridable | Used? |
|----------|-------|-------------|-------|
| `sink.rs` DEFAULT_TILE_SIZE | 512 | Yes (TileConfig) | Yes -- primary knob |
| `GpuConfig.gpu_tile_size` | 4096 | Yes (set_gpu_config) | **No -- dead code** |
| `GpuConfig.max_full_image_pixels` | 16M | Yes (set_gpu_config) | **No -- dead code** |
| `SpatialCache` | n/a | Yes (budget) | **No -- not queried in dispatch** |

### Problems

1. **One size fits all** -- 512x512 for every node regardless of compute characteristics
2. **No runtime feedback** -- tile size is static, cannot adapt to VRAM pressure or throughput
3. **Overlap waste** -- a 512x512 tile with 32px blur overlap requests 576x576 from upstream (27% waste). Larger tiles reduce this ratio.
4. **No per-subgraph control** -- export branch and preview branch get identical tile sizes
5. **Dead configuration** -- GpuConfig exists but is never consulted

### Memory Profile: 64MP Rgba32f Image (8000x8000)

| Tile Size | Pixels/tile | Bytes/tile (f32) | Overlap waste (r=16 blur) | Tiles needed |
|-----------|-------------|------------------|--------------------------|--------------|
| 256x256 | 65K | 1 MB | 23% | 961 |
| 512x512 | 262K | 4 MB | 12% | 256 |
| 1024x1024 | 1M | 16 MB | 6% | 64 |
| 2048x2048 | 4M | 64 MB | 3% | 16 |
| 4096x4096 | 16M | 256 MB | 1.5% | 4 |

Observation: for neighborhood ops, the sweet spot is 1024-2048. Below 512, overlap waste dominates. Above 2048, memory pressure dominates. Point ops are tile-agnostic.

---

## 2. Design: Layered Demand Control

Three layers, CSS-like cascade -- inner overrides outer:

```
Layer 1: Pipeline Default Strategy    (global baseline)
Layer 2: DemandRegulator Node         (subgraph override)
Layer 3: Node TileHint                (informational, advisory)
```

### Layer 1: Pipeline Default Strategy

The global baseline. Every pipeline gets this. Simple pipelines never configure anything else.

```rust
// Default: current behavior (Fixed 512)
let pipeline = Pipeline::new();

// Memory-budget-aware: adapt tile size to fit within budget
pipeline.set_demand_strategy(DemandStrategy::AdaptiveMemory {
    budget_bytes: 64 * 1024 * 1024, // 64 MB peak working set
});

// VRAM-aware: adapt GPU tile size to VRAM budget
pipeline.set_demand_strategy(DemandStrategy::AdaptiveVram {
    vram_budget_bytes: 128 * 1024 * 1024, // 128 MB GPU budget
    cpu_tile_size: 512,                     // CPU tiles stay small
});

// Explicit: full control
pipeline.set_demand_strategy(DemandStrategy::Fixed {
    cpu_tile_size: 1024,
    gpu_tile_size: 2048,
});
```

**Implementation:** `DemandStrategy` is stored on `NodeGraph`. The sink's `request_tiled` reads it instead of hardcoding 512. GPU dispatch reads it for GPU-specific tile sizing.

```rust
pub enum DemandStrategy {
    /// Current behavior: fixed tile size for everything.
    Fixed { cpu_tile_size: u32, gpu_tile_size: u32 },

    /// Adapt tile size based on total memory budget.
    /// Computes: tile_size = sqrt(budget / (bpp * pipeline_depth))
    AdaptiveMemory { budget_bytes: usize },

    /// Separate CPU and GPU budgets. GPU tiles sized to VRAM budget.
    AdaptiveVram {
        vram_budget_bytes: usize,
        cpu_tile_size: u32,
    },
}

impl Default for DemandStrategy {
    fn default() -> Self {
        DemandStrategy::Fixed {
            cpu_tile_size: 512,
            gpu_tile_size: 4096,
        }
    }
}
```

### Layer 2: DemandRegulator Node

Surgical override inserted in the graph. Controls demand for its downstream subgraph.

```rust
// "This blur chain is expensive -- use larger tiles to reduce overlap waste"
let blur_chain_end = pipeline.blur(source, 20.0);
let regulated = pipeline.add_demand_regulator(blur_chain_end, DemandHint::TileSize(2048));

// "This export branch goes to a low-memory device -- use tiny tiles"
let export = pipeline.add_demand_regulator(processed, DemandHint::MemoryBudget {
    budget_bytes: 8 * 1024 * 1024,
});

// "Let the system decide based on the node's preferred sizes"
let auto = pipeline.add_demand_regulator(processed, DemandHint::Auto);
```

**DemandRegulator is a transparent graph node:**
- Implements `ImageNode` with identity `compute_region` (passes through pixels unchanged)
- Stores a `DemandHint` that the graph walker checks during demand propagation
- Does NOT process pixels -- zero overhead in the compute path
- Only affects how `request_region` chunks requests to nodes above it

```rust
pub enum DemandHint {
    /// Fixed tile size for this subgraph.
    TileSize(u32),

    /// Adapt tile size to fit within this memory budget.
    MemoryBudget { budget_bytes: usize },

    /// Use node-reported preferred tile sizes (Layer 3 hints).
    Auto,
}

pub struct DemandRegulatorNode {
    upstream: u32,
    source_info: ImageInfo,
    hint: DemandHint,
}

impl ImageNode for DemandRegulatorNode {
    fn info(&self) -> ImageInfo { self.source_info.clone() }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // Identity -- pass through. The magic happens in request_region dispatch.
        upstream_fn(self.upstream, request)
    }

    fn upstream_id(&self) -> Option<u32> { Some(self.upstream) }
    fn access_pattern(&self) -> AccessPattern { AccessPattern::Sequential }
}
```

**Graph walker integration in `request_region`:**

```rust
// Before dispatching to GPU/CPU, check if this node IS a regulator
if let Some(regulator) = self.get_demand_regulator(node_id) {
    // Re-tile the request according to the regulator's hint
    let effective_tile_size = regulator.resolve_tile_size(bpp, pipeline_depth);
    // ... re-chunk the request into sub-tiles of effective_tile_size
    // ... dispatch each sub-tile to upstream via the regulator's identity pass-through
}
```

Alternatively (simpler): the regulator doesn't re-tile itself. Instead, when the sink is deciding tile size, it walks the graph to find the nearest DemandRegulator above each node and uses its hint. This avoids re-tiling mid-graph.

**Recommended approach:** The sink walks the graph once at write time to build a tile-size map. Each node gets an "effective tile size" from the nearest regulator (or pipeline default). The sink then tiles each subgraph independently.

### Layer 3: Node TileHint (Advisory)

Nodes declare what tile size they work best at. The strategy/regulator uses these hints to make better decisions.

```rust
pub trait ImageNode {
    // ... existing methods ...

    /// Optional: hint the demand system about preferred tile size.
    /// Neighborhood ops should report their overlap ratio at different tile sizes.
    /// Returns None for tile-agnostic nodes (point ops, most transforms).
    fn tile_hint(&self) -> Option<TileHint> {
        None  // Default: no preference
    }
}

pub struct TileHint {
    /// Minimum tile size below which this node wastes significant overlap.
    pub min_efficient_tile: u32,
    /// Kernel/overlap radius in pixels (0 for point ops).
    pub overlap_radius: u32,
}
```

**Auto strategy uses hints:** When `DemandHint::Auto` is set, the regulator collects hints from all nodes in its subgraph and picks the tile size that minimizes total overlap waste while respecting the memory budget.

---

## 3. Resolution Order (Cascade)

When `request_region` needs to determine tile size for a node:

```
1. Is there a DemandRegulator between this node and the sink?
   YES --> use the regulator's DemandHint
   NO  --> continue

2. Does the pipeline have a DemandStrategy set?
   YES --> use the strategy (Fixed/AdaptiveMemory/AdaptiveVram)
   NO  --> continue

3. Fall back to DEFAULT_TILE_SIZE (512)
```

Within each layer, `DemandHint::Auto` consults Layer 3 (node hints) to pick the optimal tile size.

---

## 4. Decision Matrix

| Criterion | Layer 1 Only | Layer 1+2 | Layer 1+2+3 |
|-----------|-------------|-----------|-------------|
| **API simplicity** | Trivial | Simple | Moderate |
| **Zero-config works** | Yes | Yes | Yes |
| **Per-subgraph control** | No | Yes | Yes |
| **Adapts to node characteristics** | No | Manual | Automatic |
| **Backward compatible** | Yes | Yes | Yes |
| **Implementation complexity** | Low (1 enum + sink change) | Medium (+1 node type + walker check) | Medium (+trait method) |

**Recommendation:** Implement in order. Layer 1 first (immediate value, minimal risk). Layer 2 when users need per-subgraph control. Layer 3 when profiling shows overlap waste is a real bottleneck.

---

## 5. Comparison: rasmcore vs Industry

| Aspect | rasmcore (proposed) | VIPS | Nuke | Halide | OIIO |
|--------|-------------------|------|------|--------|------|
| **Default behavior** | Fixed 512 tile grid | Sink-type selection | Scanline auto | Must write schedule | Must specify ROI |
| **Per-node control** | DemandRegulator node | None | Tile accessor per op | compute_at per stage | ROI per op call |
| **Global control** | DemandStrategy enum | Sink type (screen/disc) | Memory limit | Auto-scheduler | None |
| **Adaptive** | AdaptiveMemory/Vram | No | LRU row cache | Auto-scheduler | No |
| **Learning curve** | Zero for default, one method for strategy, one node for advanced | Low | Medium | High | Medium |

Our design is closest to **Nuke's hybrid model** (automatic default + per-node tuning) with **Halide's separation of concerns** (algorithm doesn't know about tiling, strategy is external). We avoid OIIO's explicitness (too much burden on users) and VIPS's inflexibility (no per-node control).

---

## 6. Implementation Tracks

### Track A: `demand-strategy-core` (Layer 1)
**Scope:** DemandStrategy enum, store on NodeGraph, sink reads it, wire GpuConfig into it.
**Tasks:** ~8
**Dependencies:** None
**Files:** gpu.rs, graph.rs, sink.rs, builder.rs

### Track B: `demand-regulator-node` (Layer 2)
**Scope:** DemandRegulatorNode, graph walker integration, pipeline.add_demand_regulator().
**Tasks:** ~10
**Dependencies:** Track A
**Files:** graph.rs, new nodes/demand_regulator.rs, builder.rs

### Track C: `demand-node-hints` (Layer 3)
**Scope:** TileHint trait method on ImageNode, Auto resolution, hint implementations on blur/sharpen/bilateral.
**Tasks:** ~8
**Dependencies:** Track B
**Files:** graph.rs, blur.rs, sharpen.rs, bilateral.rs, etc.

### Track D: `demand-adaptive-vram` (Layer 1 extension)
**Scope:** AdaptiveVram strategy with GPU VRAM probing, dynamic tile size adjustment based on executor feedback.
**Tasks:** ~6
**Dependencies:** Track A
**Files:** gpu.rs, gpu_executor.rs, graph.rs

---

## 7. Backward Compatibility

**Zero breakage guarantee:**

- `DemandStrategy::default()` = `Fixed { cpu_tile_size: 512, gpu_tile_size: 4096 }` = identical to current behavior
- No DemandRegulator nodes = no change in demand propagation
- `tile_hint()` returns `None` by default = no change in any existing node
- `TileConfig` still works = existing `write_tiled()` callers unchanged
- The `TileConfig` passed to sink is the **output** tile size. The `DemandStrategy` controls **internal** tile sizes. They can differ (sink tiles at 512 for streaming, internal processing at 2048 for efficiency).

**Migration path:**
1. Ship Track A with `Fixed` as default -- identical behavior
2. Users opt in via `pipeline.set_demand_strategy(AdaptiveMemory { ... })`
3. Ship Track B -- users insert regulators where needed
4. Ship Track C -- existing nodes get hints, `Auto` mode works
