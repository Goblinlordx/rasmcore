# V2 Pipeline Migration Guide

## Overview

The V2 pipeline (`rasmcore-pipeline-v2`) is a ground-up rebuild that replaces the V1 pipeline with a clean, f32-native architecture. V1 continues to work but is deprecated — all new development should target V2.

## What Changed

| Aspect | V1 | V2 |
|---|---|---|
| **Pixel data** | `&[u8]` + PixelFormat (20 variants) | `&[f32]` — always RGBA, 4 channels |
| **Format dispatch** | 1,621 PixelFormat references | Zero. None. |
| **GPU buffers** | U32Packed + F32Vec4 | Always f32 |
| **GPU priority** | CPU primary, GPU optional | GPU primary, CPU fallback |
| **Filter API** | `fn(Rect, UpstreamFn, ImageInfo) -> Vec<u8>` | `fn(&self, &[f32], w, h) -> Vec<f32>` |
| **Color pipeline** | Decorative ColorSpace field | Enforced per-node, auto-conversion |
| **Operation fusion** | u8 LUT composition | Expression tree IR + WGSL codegen |
| **Demand regulation** | Fixed 512 tile size | DemandStrategy + DemandRegulator nodes |

## V2 Crate Structure

```
rasmcore-pipeline-v2/src/
├── lib.rs              — crate root
├── node.rs             — Node trait, NodeInfo, InputRectEstimate
├── graph.rs            — Graph engine, GPU-primary dispatch
├── ops.rs              — Filter, GpuFilter, Encoder, Decoder, Transform, AnalyticOp
├── registry.rs         — OperationRegistration, ParamDescriptor, ParamConstraint
├── filter_node.rs      — FilterNode/GpuFilterNode wrappers, IO_F32
├── color_space.rs      — ColorSpace enum
├── color_math.rs       — Transfer functions, ACES matrices
├── color_convert.rs    — ColorConvertNode, ViewTransformNode
├── fusion.rs           — PointOpExpr, expression tree, WGSL codegen
├── staged.rs           — AnalysisNode, ParamBinding, StagedPipeline
├── demand.rs           — DemandStrategy, DemandHint
├── cache.rs            — SpatialCache for f32 tiles
├── gpu.rs              — GpuExecutor (f32-only)
├── rect.rs             — Rectangle geometry
├── hash.rs             — Content hashing
└── filters/            — V2 filter implementations
    ├── adjustment.rs   — brightness, contrast, gamma, etc.
    ├── spatial.rs      — blur, sharpen, median, etc.
    ├── color.rs        — hue, saturate, channel_mixer, etc.
    ├── enhancement.rs  — denoise, dehaze, etc.
    └── effect.rs       — noise, grain, pixelate, etc.
```

## Writing a V2 Filter

```rust
use rasmcore_pipeline_v2::{Filter, PipelineError};

struct MyFilter {
    strength: f32,
}

impl Filter for MyFilter {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        // input is f32 RGBA: [R0, G0, B0, A0, R1, G1, B1, A1, ...]
        // Process in f32. No format dispatch. No clamping.
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] *= self.strength;
            pixel[1] *= self.strength;
            pixel[2] *= self.strength;
            // Alpha unchanged
        }
        Ok(out)
    }
}
```

## V1 Deprecation

The following V1 APIs are deprecated:
- `PixelFormat` enum (in pipeline internals)
- `process_via_8bit` / `process_via_standard`
- `validate_format` / `is_16bit` / `is_float`
- `BufferFormat::U32Packed`
- `io_u32.wgsl` / `pixel_ops.wgsl` (u32 GPU fragments)
- Q16 EWA internals

These will be removed in a future version once all consumers migrate to V2.
