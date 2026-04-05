# GPU Multi-Output Pipeline Design

## Problem

The preview worker currently runs the full filter chain twice: once for the viewport display, once for scope rendering. With GPU display mode, the first run produces a GPU buffer that persists (`lastOutputBuf`) — but the scope path re-executes the entire chain from scratch via CPU, ignoring the GPU result entirely.

For a 5-filter chain with 3 scopes, this means 4x the work (1 viewport + 3 scope re-executions) when it should be 1x (compute once, distribute to 4 outputs).

## Design: ref / branch / addDisplay / execute

### Core Primitives

| Primitive | Purpose |
|-----------|---------|
| `ref(name)` | Name the current node as a branch point in the DAG |
| `branch(name)` | Move cursor to a named ref — subsequent ops fork from there |
| `addDisplay(canvas, viewport?)` | Attach a render target at the current cursor position |
| `execute()` | Run the full DAG once, blit all display targets, one `queue.submit()` |

### Fluent API

```typescript
pipeline
  .read(imageBytes)                     // node 0: source
  .brightness({ amount: 0.2 })         // node 1
  .contrast({ amount: 0.3 })           // node 2
  .addDisplay(viewportCanvas)           // display 0: blit node 2 → viewport
  .ref("graded")                        // mark node 2 as "graded"
  .branch("graded")                     // cursor → node 2
    .scopeHistogram()                   // node 3: reads from node 2
    .addDisplay(histogramCanvas)        // display 1: blit node 3 → histogram
  .branch("graded")                     // cursor → node 2 again
    .scopeWaveform()                    // node 4: reads from node 2
    .addDisplay(waveformCanvas)         // display 2: blit node 4 → waveform
  .execute()                            // one submit: compute 0→1→2→3→4, blit ×3
```

### What happens without GPU scopes

Even with CPU-only scopes, the ref/branch model eliminates re-execution:

```typescript
pipeline
  .read(imageBytes)
  .brightness({ amount: 0.2 })
  .contrast({ amount: 0.3 })
  .addDisplay(viewportCanvas)           // GPU blit
  .ref("graded")
  .branch("graded")
    .scopeHistogram()                   // CPU scope — reads from cached "graded"
    .addDisplay(histogramCanvas)        // 2D canvas blit (not GPU)
  .execute()
```

The chain (source → brightness → contrast) executes once. The scope reads from the cached result at the ref point. No GPU readback needed if the scope does its own CPU render — the ref provides the cached f32 pixels. **The win is avoiding chain re-execution, not eliminating CPU readback.**

With GPU scope shaders (future), the scope would also be a compute pass in the same submit, eliminating CPU readback entirely.

## WebGPU Feasibility

### Multi-canvas from single device: PROVEN

The codebase already does this. `GpuHandlerV2` has:
- `canvasCtx` (processed view) + `origCtx` (original view)
- Both share the same `GPUDevice` and `blitPipeline`
- Each has its own `GPUCanvasContext`, viewport uniform buffer, and pixel source buffer
- `blitOriginal()` reuses the blit render pipeline with a different bind group

**Adding scope canvases is the same pattern** — one more `GPUCanvasContext` per scope, reusing the existing blit pipeline.

### Buffer sharing rules

- A `GPUBuffer` with `STORAGE` usage can be bound as `read-only-storage` in multiple bind groups
- Multiple render passes in one command encoder can read from the same buffer
- Buffer must not be destroyed until after `queue.submit()` completes
- Existing `deferDestroy()` pattern handles this

### Constraints

- `getCurrentTexture()` returns a new texture each frame — must be called per canvas per submit
- Each canvas needs its own viewport uniform buffer (different pan/zoom/dimensions)
- Maximum bind groups per pipeline: 4 (WebGPU limit) — not a concern for blit (uses 1)

## Architecture

### Rust Side: Graph refs

The V2 Graph already supports DAG topology — nodes can have multiple downstream consumers. What's missing is **named access points**.

```rust
// New fields on Graph
refs: HashMap<String, u32>,          // ref name → node_id

// New methods
pub fn set_ref(&mut self, name: &str, node_id: u32) {
    self.refs.insert(name.to_string(), node_id);
}

pub fn get_ref(&self, name: &str) -> Option<u32> {
    self.refs.get(name).copied()
}
```

The `branch()` operation in the fluent SDK just calls `get_ref()` and updates the cursor node. No new WIT types needed — `ref` and `branch` map to existing node IDs.

### WIT Surface

```wit
// New methods on image-pipeline-v2:
set-ref: func(name: string, node: node-id);
get-ref: func(name: string) -> option<node-id>;

// Multi-output GPU plan
record display-target {
    node-id: node-id,
    target-name: string,     // "viewport", "histogram", "waveform", etc.
}

render-multi-gpu-plan: func(targets: list<display-target>)
    -> result<multi-gpu-plan, rasmcore-error>;

record multi-gpu-plan {
    stages: list<gpu-stage>,
}

record gpu-stage {
    target-name: string,             // which display this serves
    shaders: list<gpu-shader>,       // compute chain for this branch
    input-source: stage-input,
    width: u32,
    height: u32,
}

variant stage-input {
    pixels(pixel-buffer),            // upload from CPU (source node)
    prior-stage(string),             // read from a prior stage's output buffer
}
```

The host extracts the multi-plan, then executes stages in order. When a stage's input is `prior-stage("viewport")`, the host binds the GPU buffer from that completed stage instead of uploading new pixels.

### JS Side: GpuHandler Multi-Display

```typescript
interface DisplayRegistration {
  name: string;
  canvas: OffscreenCanvas;
  ctx: GPUCanvasContext;
  viewportBuf: GPUBuffer;
}

class GpuHandlerV2 {
  private displays: Map<string, DisplayRegistration> = new Map();

  // Register a named display target
  addDisplay(name: string, canvas: OffscreenCanvas, hdr: boolean): void;

  // Execute multi-stage plan — one queue.submit()
  async executeMulti(plan: MultiGpuPlan): Promise<GpuError | null>;
}
```

`executeMulti()`:
1. Create one `GPUCommandEncoder`
2. For each stage:
   a. Resolve input: `pixels` → create + write buffer, `prior-stage` → reuse buffer from prior stage
   b. Execute compute passes (ping-pong, same as current `executeAndDisplay`)
   c. If stage has a display target: `appendBlitPass()` to that target's canvas
   d. Store output buffer by stage name for downstream stages
3. `device.queue.submit([encoder.finish()])`
4. Destroy intermediate buffers (keep display-targeted ones for `displayOnly()`)

### Fluent SDK

```typescript
class Pipeline {
  private _pipe: RawPipeline;
  private _node: number;
  private _displays: Map<string, { nodeId: number; canvas: OffscreenCanvas }>;
  private _refs: Map<string, number>;

  ref(name: string): Pipeline {
    this._pipe.setRef(name, this._node);
    this._refs.set(name, this._node);
    return this;
  }

  branch(name: string): Pipeline {
    const nodeId = this._refs.get(name);
    if (nodeId == null) throw new Error(`Unknown ref: ${name}`);
    // Return a new Pipeline cursor at the ref node
    const branched = Object.create(Pipeline.prototype);
    branched._pipe = this._pipe;
    branched._node = nodeId;
    branched._displays = this._displays;
    branched._refs = this._refs;
    return branched;
  }

  addDisplay(name: string, canvas: OffscreenCanvas): Pipeline {
    this._displays.set(name, { nodeId: this._node, canvas });
    return this;
  }

  async execute(gpuHandler: GpuHandlerV2): Promise<void> {
    const targets = [...this._displays.entries()].map(([name, d]) => ({
      targetName: name,
      nodeId: d.nodeId,
    }));
    const plan = this._pipe.renderMultiGpuPlan(targets);
    if (plan) {
      // Register canvases with GPU handler
      for (const [name, d] of this._displays) {
        if (!gpuHandler.hasDisplay(name)) {
          gpuHandler.addDisplay(name, d.canvas, isHdrDisplay());
        }
      }
      await gpuHandler.executeMulti(plan);
    }
  }
}
```

## Buffer Lifecycle

### At ref branch points

```
Stage: source → brightness → contrast [ref "graded"]
  bufA: source pixels (uploaded)
  bufB: brightness output (ping)
  bufA: contrast output (pong) ← this is "graded" output
  
  bufA is NOT destroyed — it's referenced by downstream stages

Stage: branch "graded" → histogram
  Input: bufA (prior-stage "graded")
  bufC: histogram compute (new allocation, scope output is 512x512)
  Blit bufC → histogram canvas
  bufC destroyed (or kept for displayOnly re-blit)

Stage: branch "graded" → waveform
  Input: bufA (prior-stage "graded" — still alive)
  bufC: reused for waveform compute (same 512x512 size)
  Blit bufC → waveform canvas
  bufC destroyed

After all stages: bufA kept as lastOutputBuf for viewport re-blit
```

### Memory budget

| Buffer | Size | Lifetime |
|--------|------|----------|
| Source upload | W×H×16 | Until first compute reads it |
| Ping-pong pair | W×H×16 each | Reused across chain |
| Ref output | W×H×16 | Until all branches complete |
| Scope ping-pong | 512×512×16 | Reused across scope branches |
| Display output | varies | Kept for `displayOnly()` re-blit |

For a 1920×1080 image with 3 scopes: ~32MB ref buffer + ~4MB scope buffers = ~36MB peak. Well within GPU memory budgets.

## Performance Model

### Current (re-execution)

```
processChain:
  chain execution:   ~15ms (GPU compute)
  viewport blit:     ~1ms
processScope (×3):
  chain execution:   ~15ms × 3 = 45ms (CPU re-execution!)
  scope compute:     ~5ms × 3 = 15ms (CPU)
  PNG encode:        ~3ms × 3 = 9ms
  postMessage:       ~1ms × 3

Total: ~86ms per frame
```

### With multi-output (CPU scopes, cached ref)

```
execute:
  chain execution:   ~15ms (GPU compute, once)
  viewport blit:     ~1ms
  scope compute:     ~5ms × 3 = 15ms (CPU, reads cached f32)
  scope canvas blit: ~1ms × 3

Total: ~34ms per frame (2.5x faster)
```

### With multi-output + GPU scopes (future)

```
execute (single submit):
  chain compute:     ~15ms
  scope compute:     ~0.5ms × 3 (GPU, reads from same buffer)
  all blits:         ~1ms × 4

Total: ~20ms per frame (4.3x faster)
```

## Migration Path

### Phase 1: Graph refs (Rust, minimal)
- Add `refs: HashMap<String, u32>` to Graph
- Add `set_ref()` / `get_ref()` methods
- Add WIT methods: `set-ref`, `get-ref`
- **No breaking changes** — existing single-output still works

### Phase 2: Multi-display GpuHandler (JS)
- Add `displays: Map<string, DisplayRegistration>`
- `addDisplay(name, canvas, hdr)` method
- Refactor existing `canvasCtx`/`origCtx` into the display map
- `executeMulti(plan)` method alongside existing `executeAndDisplay()`

### Phase 3: Multi-GPU plan (Rust + WIT)
- `render-multi-gpu-plan(targets)` returns ordered stages
- Each stage specifies input source: `pixels` or `prior-stage`
- Topological sort ensures correct execution order

### Phase 4: Fluent SDK (TypeScript)
- Add `ref()`, `branch()`, `addDisplay()`, `execute()` to Pipeline class
- `execute()` replaces the current `write('png')` + manual GPU dispatch

### Phase 5: Worker integration
- Replace `processChain()` + `processScope()` with single `execute()`
- Preview worker manages display registrations
- Main thread sends chain + scope config in one message

### Phase 6 (future): GPU scope shaders
- Implement histogram/waveform/vectorscope as WGSL compute shaders
- These become stages in the multi-plan alongside the filter chain
- Eliminates CPU readback entirely

## Backward Compatibility

- `render-gpu-plan(node)` remains — it's the single-output path
- `render-multi-gpu-plan(targets)` is additive
- Existing `executeAndDisplay()` unchanged — `executeMulti()` is new
- `write()` and `render()` still work for CPU/encode paths
- Workers can adopt incrementally: use multi-output for GPU display mode, fall back to current path otherwise
