# Multi-Pass Analysis-Then-Render Pipeline Design

## The Critical Question: GPU-Only Analysis

Analysis must NOT force CPU roundtrips. The current multi-pass GPU shader
infrastructure already solves this for single-filter analysis (histogram →
apply in equalize). The design challenge is extending this to **cross-node**
analysis where node A's GPU analysis configures node B's GPU render.

## Current Pass Model

### Backwards Pass (Implicit — ROI Negotiation)
```
request_region(sink, tile) → each node returns input_rect() →
  Exact(expanded_rect)  — spatial ops add kernel overlap
  FullImage             — tile barrier, forces full materialization
  UpperBound(rect)      — conservative estimate
```
This happens naturally through the recursive request model. No explicit
backwards walk — each node declares what it needs when asked.

### Forwards Pass (Implicit — Pixel Execution)
```
source.compute(rect) → filter1.compute(rect) → filter2.compute(rect) → sink
```
Data flows forward through the chain. GPU dispatch batches consecutive
GPU-capable nodes into a single compute chain (`collect_gpu_chain`).

### GPU Chain Collection (Backwards)
```
collect_gpu_chain(sink_id):
  walk backwards from sink
  collect gpu_shaders() from each node with 1 upstream
  stop at first non-GPU node
  return (source_id, [shader1, shader2, ...])
```
The entire chain executes as one GPU submit: ping-pong buffers, no
CPU intermediates. **This is the zero-copy render path.**

## The Problem: Analysis Breaks the GPU Chain

An analysis node between two GPU-capable render nodes breaks the chain:

```
source → [GPU] brightness → [ANALYSIS] histogram → [GPU] auto_level
                                 ↑
                    GPU chain breaks here — analysis
                    needs full image, produces a scalar,
                    not pixels. Can't ping-pong through it.
```

Current behavior: `collect_gpu_chain` stops at the analysis node (no
gpu_shaders()). Two separate GPU chains form. CPU materializes the
intermediate result.

## Solution: Three-Phase GPU Execution

### Phase 1: Analysis Dispatch (GPU)
- Run analysis compute shaders on the full image
- Produce small results (histogram buffer, scalar uniforms, ROI rect)
- Results stay on GPU as buffers/uniforms

### Phase 2: Bind (CPU — tiny, O(1))
- Read back analysis results (small: 256 u32s for histogram, 4 floats for ROI)
- Map to downstream node params
- This is the ONLY CPU roundtrip and it's tiny (bytes, not megapixels)

### Phase 3: Render Dispatch (GPU)
- Execute the render chain with analysis results injected as uniforms
- Full GPU chain — no image data leaves the GPU

### Why Phase 2 is Acceptable
Reading back 256 u32s (histogram) or 4 floats (crop rect) is ~1KB.
The GPU→CPU→GPU roundtrip for this is <1ms. The alternative (keeping
the analysis result on GPU and passing it as a buffer to the render
shader) is possible but adds complexity for negligible gain.

For the **future GPU-only path**: analysis results can stay as GPU
buffers and be bound directly to render shader uniforms. This
eliminates even the tiny readback. The infrastructure for this exists
via `ReductionBuffer` — persistent buffers that chain across passes.

## Interaction with Tile-Based Execution

### Current Tile Model
```
request_tiled(sink, tile_size=512):
  for each 512x512 tile:
    request_region(sink, tile_rect)
    → walks upstream, each node tiles independently
    → FullImage nodes force full materialization at that point
```

### Analysis Nodes as Tile Barriers
An analysis node that needs the full image (histogram, face detection,
content-aware crop) is a **tile barrier**:

```
tiles → brightness → [BARRIER: histogram analysis] → auto_level → tiles
                          ↑                              ↓
              full image materializes here      tiling resumes here
```

**Key insight**: tile barriers already work via `InputRectEstimate::FullImage`.
The demand-driven model handles this naturally:
1. Downstream tile request hits the analysis node
2. Analysis node requests `FullImage` from upstream
3. Full image materializes (one-time cost, cached)
4. Analysis runs (GPU compute on full image)
5. Result bound to downstream params
6. Downstream tiling resumes with resolved params

### Memory Impact
Each tile barrier materializes the full image:
- 1920×1080 × 4 channels × 4 bytes = ~32MB per barrier
- 3 analysis nodes in chain = ~96MB peak (concurrent barriers)
- Spatial cache reuse: if upstream hasn't changed, cache hit

### Recursive Analysis Chains
```
source → [ANALYSIS] face_detect → crop → emboss → [ANALYSIS] histogram → auto_level
```
Two barriers, each materialized independently:
1. face_detect: full image → crop rect (small readback)
2. histogram: full (cropped) image → histogram (256 u32 readback)

Caching makes repeated analysis on unchanged data free (content-hash hit).

## Professional Tool Comparison

### Passes We Should Support

| Pass | Status | Purpose |
|------|--------|---------|
| **Analysis** | PARTIAL (staged.rs types, no execute()) | Histogram, face, energy, statistics |
| **Bind** | PARTIAL (ParamBinding types, no integration) | Map analysis results to node params |
| **Render** | DONE (Graph.request_full) | Demand-driven pixel execution |
| **ROI Negotiate** | DONE (InputRectEstimate) | Each node declares input region |
| **Convergence** | DONE (convergence_check) | Iterative algorithms stop when converged |
| **Multi-resolution** | NOT DONE | Gaussian/Laplacian pyramids |
| **Hash/Change** | PARTIAL (content_hash) | Skip unchanged nodes |

### DaVinci Resolve Model
- **ROI negotiate** (backwards) → we have via `input_rect()`
- **Render** (forwards) → we have via `request_region`
- **Pre-computed analysis** → we need: `StagedPipeline.execute()`
- **Tile-based render** → we have via `request_tiled`

### Nuke Model
- **Hash pass** → we have partially via `content_hash` in layer cache
- **ROI pass** → we have via `InputRectEstimate`
- **Render pass** → we have

## What Needs to Change

### Minimal Changes (Incremental)
1. **Complete `StagedPipeline.execute()`** — orchestrate analyze → bind → render
2. **Fix `add_analysis_node()` ownership** — store analysis metadata alongside graph node
3. **Wire AnalysisNode trait into existing filters** — Otsu, auto_level, equalize can declare analysis phase

### No Changes Needed
- Graph execution model (already handles FullImage barriers)
- GPU dispatch path (collect_gpu_chain works with barriers)
- Tile-based execution (demand-driven, barriers natural)
- Caching (content-hash, spatial cache — all work with analysis)
- Node trait (input_rect already supports FullImage)

### Future GPU-Only Path
- Analysis results as `ReductionBuffer` passed to render shaders
- Analysis GPU compute + render GPU compute in single submit
- Eliminates the tiny param readback
- Requires: multi-stage GPU plan (already implemented in multi-output track)

## Implementation Tracks

### Track 1: Complete StagedPipeline (small, focused)
- Implement `execute()`: analyze → bind → render
- Fix `add_analysis_node()` ownership model
- Test with histogram analysis → auto_level binding
- ~8 tasks

### Track 2: Analysis Node Implementations (parallel)
- HistogramAnalysis → ScalarVec (256 f32 bins)
- ContentAwareROI → Rect (smart crop region)
- AutoExposureAnalysis → Scalar (EV value)
- AutoWhiteBalanceAnalysis → Matrix (color correction)
- ~10 tasks

### Track 3: Migrate Existing Filters (after Track 1)
- Otsu threshold → HistogramAnalysis + threshold_binary binding
- Auto_level → MinMaxAnalysis + levels binding
- Equalize → HistogramAnalysis + CDF binding
- ~6 tasks

### Track 4: GPU Analysis Dispatch (after Track 1 + 2)
- Analysis as GPU reduction passes (histogram already works)
- Analysis result as uniform buffer for render shader
- Single GPU submit: analysis compute → render compute
- ~8 tasks
