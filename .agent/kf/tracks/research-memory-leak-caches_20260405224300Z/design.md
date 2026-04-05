# Memory Leak Audit — All Cache Systems

**Track:** research-memory-leak-caches_20260405224300Z
**Date:** 2026-04-06

---

## Executive Summary

Multiple cache systems have no eviction or cleanup lifecycle. The most critical:
**LayerCache cleanup is never called in web-ui workers** — old image filter
results accumulate indefinitely. Combined with unbounded BufferPool growth
and a FIFO index corruption bug in SpatialCache, memory grows monotonically
during interactive use.

---

## 1. LayerCache — CRITICAL (no cleanup called)

**Location:** `crates/rasmcore-pipeline-v2/src/layer_cache.rs`

**What it caches:** f32 RGBA pixel data, keyed by blake3 content hash.
Cross-pipeline — persists across pipeline rebuilds for the same filter chain.

**Budget:** Starts at 64MB (web-ui worker), can auto-grow to 1GB max.

**The leak:** `cleanup_unreferenced()` (line 209-219) exists and correctly
frees entries, but **it is never called** from the web-ui workers:
- `v2-preview-worker.ts` — no cleanup calls
- `v2-pipeline-worker.ts` — no cleanup calls
- `graph.rs` has `cleanup_layer_cache()` (line 212) — **zero callers**

**Expected lifecycle (not implemented):**
```
reset_references() → processChain → cleanup_unreferenced()
```

**Actual lifecycle:**
```
processChain → processChain → processChain → ... (entries accumulate forever)
```

**Impact:** Loading image A, adjusting filters → cached. Loading image B →
image A's filter results stay in cache. After 10 images, cache holds results
for all 10. Budget doubles under pressure up to 1GB.

**Fix:** Call `reset_references()` before each `processChain()` and
`cleanup_unreferenced()` after. In the WASM adapter, expose these on
the pipeline resource. In the worker, call them around each render cycle.

---

## 2. SpatialCache FIFO — CRITICAL (index corruption bug)

**Location:** `crates/rasmcore-pipeline-v2/src/cache.rs` lines 74-81

**The bug:** FIFO eviction uses `entries.remove(evict_idx)` which shifts
all subsequent entries down by one. But `self.order` stores `(node_id, entry_index)`
tuples with the **original** index. After a removal, stored indices are stale:

```rust
// order has (node_1, 0), (node_1, 1), (node_1, 2)
// evict index 0 → entries shifts: [B, C] at indices [0, 1]
// but order still has (node_1, 1) and (node_1, 2) — wrong!
```

**Impact:** Entries at wrong indices may be evicted (data corruption) or
entries may be skipped entirely (never evicted → leak). Can also panic
if `evict_idx >= entries.len()`.

**Fix:** After removal, update all indices in `order` that reference the
same node. Or switch to a different eviction strategy (e.g., generation
counter, or just clear the entire cache on budget overflow).

---

## 3. BufferPool — HIGH (unbounded, no max size)

**Location:** `crates/rasmcore-pipeline-v2/src/graph.rs` lines 26-55

**What it caches:** Reusable `Vec<f32>` pixel buffers.

**The leak:** No maximum pool size. Buffers retain their peak allocated
capacity forever (`resize` grows but never shrinks). Pool accumulates
buffers across concurrent pipeline executions.

**Scenario:**
1. Process 4K image → buffer allocated at 33MB
2. Switch to 720p image → buffer capacity stays 33MB
3. Pool now holds an oversized buffer permanently

**On SourceResource:** Each Source has its own BufferPool. Loading a new
image creates a new Source + new pool. The old Source's pool is freed
by GC only if no references remain — but the graph may hold a reference
via `set_buffer_pool()`.

**Fix:**
- Add `max_idle: usize` to BufferPool — drop excess buffers on release
- Call `shrink_to_fit()` on buffers when returning to pool if capacity
  exceeds 2x the requested size
- Clear the buffer pool reference in Graph on drop (already done via
  `buffer_pool: Option<Rc<...>>` — drops when Graph drops)

---

## 4. SpatialCachePool — MEDIUM (no max idle count)

**Location:** `crates/rasmcore-pipeline-v2/src/cache.rs` lines 128-181

**What it caches:** Reusable SpatialCache instances (HashMap + Vec containers).

**The leak:** Pool grows to peak concurrent pipeline count and never shrinks.
Each cache holds HashMap/Vec allocations (~few KB empty, but capacity from
previous use may be much larger).

**Impact:** Low in practice — web-ui uses 1 pipeline at a time, so pool
stays at size 1. Only a concern for concurrent hosts.

**Fix:** Add `max_idle` parameter. On `release()`, if pool already has
`max_idle` caches, drop the new one instead of keeping it.

---

## 5. GPU Shader Caches — LOW (bounded by filter count)

### Rust: WgpuExecutorV2

**Location:** `crates/rasmcore-cli/src/gpu_executor_v2.rs` line 29

**What:** `HashMap<u64, wgpu::ShaderModule>` — source hash → compiled module.

**Growth:** One entry per unique WGSL shader source. Bounded by the number
of distinct filter GPU shaders (~20-30 in the registry). FusedPointOpNode
shaders vary by expression but the LUT-based CPU path handles most cases.

**Cleared:** Never explicitly. Freed on executor drop.

**Risk:** Low — shader count is finite.

### TypeScript: GpuHandlerV2.shaderCache

**Location:** `web-ui/src/gpu-handler-v2.ts` line 54 (also `sdk/v2/lib/gpu-handler.ts`)

**What:** `Map<string, GPUComputePipeline>` — hash+entryPoint → pipeline.

**Growth:** Same as Rust — bounded by unique shader sources.

**Cleared:** On `device.lost` event (line 94). Also on `destroy()`.

**Risk:** Low.

---

## 6. Docs sourceCache — LOW

**Location:** `docs/site/src/lib/wasm-loader.ts` line 18

**What:** `Map<string, Source>` — reference image URL → decoded Source.

**Growth:** One entry per unique playground reference image. In practice
~1-3 entries per page visit.

**Cleared:** Never (module-scoped). Freed on page navigation.

**Risk:** Negligible.

---

## Priority-Ordered Fix Plan

| Priority | Issue | Fix | Effort |
|----------|-------|-----|--------|
| P0 | LayerCache cleanup never called | Add reset/cleanup cycle to worker | Small |
| P0 | SpatialCache FIFO index corruption | Fix index tracking or switch to clear-on-overflow | Medium |
| P1 | BufferPool unbounded growth | Add max_idle + shrink_to_fit | Small |
| P2 | SpatialCachePool no max idle | Add max_idle parameter | Small |
| P3 | GPU shader caches | No action needed (bounded) | None |

---

## Recommended Implementation Tracks

1. **fix-layer-cache-cleanup** — Add `reset_references()` + `cleanup_unreferenced()`
   calls in preview worker around processChain. ~4 tasks.

2. **fix-spatial-cache-eviction** — Fix FIFO index corruption. Either track
   indices correctly or simplify to clear-all-on-overflow. ~3 tasks.

3. **fix-buffer-pool-bounds** — Add max_idle to BufferPool + SpatialCachePool,
   shrink oversized buffers. ~4 tasks.
