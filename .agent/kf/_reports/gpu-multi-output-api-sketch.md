# GPU Multi-Output Pipeline — API Sketch

## Core Concepts

**ref(name)** — marks the current node as a named branch point in the DAG. Downstream consumers can fork from this point.

**branch(name)** — switches the "cursor" to a previously named ref. Subsequent operations build from that point.

**addDisplay(canvas)** — attaches a render target (OffscreenCanvas) at the current cursor position. The GPU result at this node will be blitted to that canvas during `execute()`.

**execute()** — runs the entire DAG in one GPU submit. All compute passes chain via ping-pong buffers. All display targets get blitted from their respective node's output buffer.

## Fluent API Example

```typescript
pipeline
  .read(imageBytes)                     // source node (node 0)
  .brightness({ amount: 0.2 })         // node 1
  .contrast({ amount: 0.3 })           // node 2
  .addDisplay(viewportCanvas)           // blit node 2 → viewport
  .ref("graded")                        // name this point
  .branch("graded")                     // fork from "graded"
    .scopeHistogram()                   // node 3 (512x512 histogram)
    .addDisplay(histogramCanvas)        // blit node 3 → histogram panel
  .branch("graded")                     // fork again from "graded"
    .scopeWaveform()                    // node 4 (512x512 waveform)
    .addDisplay(waveformCanvas)         // blit node 4 → waveform panel
  .branch("graded")                     // fork again
    .scopeVectorscope()                 // node 5
    .addDisplay(vectorscopeCanvas)      // blit node 5 → vectorscope panel
  .execute()                            // one GPU submit: 5 computes + 4 blits
```

## What `execute()` Does

1. **Topological sort** — orders nodes by dependency (source → filters → scopes)
2. **GPU plan extraction** — collects shader chains for each connected subgraph
3. **Compute passes** — ping-pong buffers through the chain, forking at ref points
4. **Blit passes** — for each `addDisplay` target, append a blit render pass from that node's output buffer to the canvas texture
5. **Single `device.queue.submit()`** — all passes in one command encoder

## Buffer Management at Branch Points

When a ref is consumed by multiple branches:
- The ref node's output buffer is **not destroyed** after the first consumer
- It stays alive until all branches have read from it
- Each branch gets its own ping-pong pair for downstream compute
- Memory: at most `N_branches + 2` buffers alive simultaneously

```
source → brightness → contrast → [ref "graded"]
                                    ├── histogram → [display]
                                    ├── waveform  → [display]
                                    └── vectorscope → [display]

Buffers at peak:
  - "graded" output (shared, read-only by all branches)
  - histogram ping-pong pair (2 buffers)
  - waveform ping-pong pair (reuse after histogram done)
  - vectorscope ping-pong pair (reuse after waveform done)
```

Sequential branch execution (histogram → waveform → vectorscope) means only 3 buffers needed at peak: the ref buffer + one ping-pong pair reused across branches.

## Display Targets

Each `addDisplay(canvas)` creates:
- A `GPUCanvasContext` on the OffscreenCanvas
- A blit render pass reading from the node's output buffer
- Viewport uniforms for that canvas (independent pan/zoom per target)

Multiple displays on the same node (e.g., viewport + thumbnail) share the same buffer — just different viewport uniforms and canvas sizes.

## WIT Surface (Rust side)

```wit
// New methods on image-pipeline-v2:
ref: func(name: string);
branch: func(name: string) -> result<node-id, rasmcore-error>;

// Execution returns a plan for ALL branches
render-multi-gpu-plan: func() -> result<multi-gpu-plan, rasmcore-error>;

record multi-gpu-plan {
    // Ordered list of compute stages
    stages: list<gpu-stage>,
}

record gpu-stage {
    node-id: node-id,
    shaders: list<gpu-shader>,
    input-source: stage-input,
    width: u32,
    height: u32,
}

variant stage-input {
    pixels(pixel-buffer),   // source node — upload from CPU
    ref-buffer(string),     // named ref — read from GPU buffer
}
```

The host (JS) executes stages in order. When `input-source` is `ref-buffer("graded")`, the host binds the previously-computed buffer instead of uploading new pixels.

## Migration Path

1. **Phase 1**: Add `ref()` and `branch()` to the graph — DAG already supports multiple downstream nodes per upstream, just needs named access points
2. **Phase 2**: Add `addDisplay()` to JS GpuHandler — register multiple canvas contexts with the handler
3. **Phase 3**: Implement `render-multi-gpu-plan` — returns ordered stages with ref dependencies
4. **Phase 4**: Implement multi-stage execution in GpuHandler — sequential branch execution with buffer reuse
5. **Phase 5**: Wire into preview worker — replace processChain + processScope with single execute()

## Key Constraint

Scopes MUST be GPU compute shaders for this to work end-to-end without CPU readback. The current scope filters are CPU-only (`ScopeNode::compute()` returns `Vec<f32>`). They need GPU shader implementations that read from a storage buffer and write to another storage buffer (which then gets blitted to the scope canvas).

This is the prerequisite: **GPU scope shaders** before multi-output can eliminate the CPU roundtrip.
