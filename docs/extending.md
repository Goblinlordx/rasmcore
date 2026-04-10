# Extending rasmcore

This guide explains how to add custom filters, codecs, transforms, and
other operations to rasmcore using the V2 architecture.

---

## Table of Contents

1. [Extension Model Overview](#extension-model-overview)
2. [Adding a Filter (V2)](#adding-a-filter-v2)
3. [Parameter Hints](#parameter-hints)
4. [GPU Filters](#gpu-filters)
5. [Trait Composition](#trait-composition)
6. [Mask Generators and Selective Adjustments](#mask-generators-and-selective-adjustments)
7. [Adding Generators, Compositors, and Mappers](#adding-generators-compositors-and-mappers)
8. [Adding a Decoder](#adding-a-decoder)
9. [Adding an Encoder](#adding-an-encoder)
10. [Pipeline Integration](#pipeline-integration)
11. [Generated Files — DO NOT EDIT](#generated-files--do-not-edit)
12. [Reference Validation](#reference-validation)
13. [PRNG Requirement](#prng-requirement)

---

## Extension Model Overview

rasmcore uses a **derive macro + trait implementation** architecture (V2):

1. Annotate a struct with `#[derive(rasmcore_macros::Filter)]` and `#[filter(...)]`
2. Implement `CpuFilter` for your struct (required)
3. Optionally implement `GpuFilter`, `PointOp`, `ColorOp`, or `AnalyticOp`
4. The codegen pipeline generates WIT declarations, SDK methods, CLI dispatch,
   pipeline nodes, and a parameter manifest

```
Source Code                    Codegen (build.rs)              Generated Output
─────────────                  ──────────────────              ────────────────
#[derive(Filter)]     ──parse──>  CodegenData    ──generate──>  pipeline nodes
#[filter(...)]                                                  SDK methods
impl CpuFilter                                                 CLI dispatch
impl GpuFilter                                                 WIT declarations
                                                                param manifest
```

### V1 vs V2 Pattern

| Aspect | V1 (deprecated) | V2 (current) |
|--------|-----------------|--------------|
| Registration | `#[register_filter]` on bare function | `#[derive(Filter)]` + `#[filter(...)]` on struct |
| Parameters | Separate `#[derive(ConfigParams)]` struct | Struct fields ARE the params |
| Implementation | Bare function `fn(Rect, UpstreamFn, ImageInfo, Config)` | `impl CpuFilter for MyFilter` |
| Config access | `config.field` | `self.field` |
| GPU support | Separate `GpuCapable` trait | `impl GpuFilter for MyFilter` |
| Optimization | `LutPointOp` trait | `PointOp`, `ColorOp`, `AnalyticOp` traits |

---

## Adding a Filter (V2)

### Step 1: Define the filter struct

Create a file in `crates/rasmcore-image/src/domain/filters/<category>/<name>.rs`:

```rust
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Invert colors with adjustable strength.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "invert_v2", category = "adjustment", reference = "color inversion")]
pub struct InvertV2 {
    /// Strength of inversion (1.0 = full invert, 0.0 = no change)
    #[param(min = 0.0, max = 1.0, step = 0.1, default = 1.0)]
    pub strength: f32,
}
```

The `#[derive(Filter)]` macro generates registration, `Default`, and parameter
metadata. Each field decorated with `#[param(...)]` becomes a UI-configurable
parameter.

### Step 2: Implement CpuFilter

```rust
impl CpuFilter for InvertV2 {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let pixels = upstream(request)?;

        // Handle f32 formats natively (V2 pipeline uses Rgba32f)
        if matches!(info.format, PixelFormat::Rgba32f | PixelFormat::Rgb32f) {
            let ch = if info.format == PixelFormat::Rgba32f { 4 } else { 3 };
            let mut samples: Vec<f32> = pixels
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            for chunk in samples.chunks_exact_mut(ch) {
                let color_ch = if ch == 4 { 3 } else { ch };
                for s in &mut chunk[..color_ch] {
                    *s = *s * (1.0 - self.strength) + (1.0 - *s) * self.strength;
                }
            }
            return Ok(samples.iter().flat_map(|v| v.to_le_bytes()).collect());
        }

        // u8 fallback path
        let mut output = pixels;
        for &mut ref mut v in output.iter_mut() {
            let orig = *v;
            let inv = 255 - orig;
            *v = ((orig as f32 * (1.0 - self.strength) + inv as f32 * self.strength) + 0.5) as u8;
        }
        Ok(output)
    }
}
```

### Step 3: Register in the module

Add to `crates/rasmcore-image/src/domain/filters/<category>/mod.rs`:

```rust
mod my_filter;
pub use my_filter::*;
```

### Step 4: Add tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::filter_traits::CpuFilter;

    #[test]
    fn invert_v2_default_is_full_invert() {
        let filter = InvertV2::default();
        let result = filter.compute(
            Rect::new(0, 0, 1, 1),
            &mut |_| Ok(vec![100, 150, 200, 255]),
            &ImageInfo { width: 1, height: 1, format: PixelFormat::Rgba8, .. },
        ).unwrap();
        assert_eq!(result, vec![155, 105, 55, 255]);
    }
}
```

### Filter attributes

| Attribute   | Required | Description |
|-------------|----------|-------------|
| `name`      | Yes      | Unique identifier used in WIT, SDK, and CLI |
| `category`  | Yes      | Grouping: `adjustment`, `spatial`, `color`, `distortion`, `effect`, `enhancement`, `morphology`, `edge`, `threshold`, `draw`, `mask`, `grading`, `tonemapping`, `analysis`, `tool`, `advanced`, `transform` |
| `group`     | No       | Sub-group for UI (e.g., `blur`, `denoise`) |
| `variant`   | No       | Variant name within the group |
| `reference` | No       | Algorithm attribution (e.g., `"Canny 1986"`) |

### Param attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `min`     | No       | Minimum value |
| `max`     | No       | Maximum value |
| `step`    | No       | Step increment |
| `default` | Yes      | Default value (used by `Default` impl) |
| `hint`    | No       | UI rendering hint (see [Parameter Hints](#parameter-hints)) |

---

## Parameter Hints

Hints control how the web UI renders filter parameters. They are metadata
only and do not affect runtime behavior.

| Hint | UI Control | Use For |
|------|-----------|---------|
| `rc.pixels` | Number spinner | Pixel dimensions, coordinates |
| `rc.log_slider` | Logarithmic slider | Radius, sigma (fine control at low end) |
| `rc.signed_slider` | Bipolar slider (centered at 0) | Offsets, shifts |
| `rc.angle_deg` | Angle dial | Rotation, hue angles |
| `rc.toggle` | Toggle switch | Boolean flags |
| `rc.seed` | Number input | Random seeds |
| `rc.temperature_k` | Gradient slider | Color temperature |
| `rc.color_rgb` | Color picker | RGB colors |
| `rc.color_rgba` | Color picker + alpha | RGBA colors |
| `rc.enum` | Dropdown select | Mode/method selection |
| `rc.text` | Text input | String parameters |
| `rc.point` | Canvas point selector | Click-to-place coordinates (cx/cy pairs) |
| `rc.path` | Canvas path drawer | Brush stroke coordinates |
| `rc.box_select` | Canvas rectangle drag | Crop/selection regions |

Example:

```rust
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "draw_circle", category = "draw")]
pub struct DrawCircleParams {
    #[param(min = 0.0, max = 65535.0, step = 1.0, default = 50.0, hint = "rc.point")]
    pub cx: f32,
    #[param(min = 0.0, max = 65535.0, step = 1.0, default = 50.0, hint = "rc.point")]
    pub cy: f32,
    #[param(min = 1.0, max = 65535.0, step = 1.0, default = 25.0, hint = "rc.log_slider")]
    pub radius: f32,
}
```

---

## GPU Filters

Filters can provide GPU acceleration via WGSL compute shaders. Implement
the `GpuFilter` trait alongside `CpuFilter`:

```rust
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "wave", category = "distortion", reference = "sinusoidal wave displacement")]
pub struct WaveParams {
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 10.0)]
    pub amplitude: f32,
    #[param(min = 1.0, max = 500.0, step = 5.0, default = 50.0)]
    pub wavelength: f32,
}

impl GpuFilter for WaveParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(
        &self, width: u32, height: u32,
        buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use rasmcore_gpu_shaders as shaders;

        // Load shader with bilinear sampling helpers
        static WAVE_F32: std::sync::LazyLock<String> =
            std::sync::LazyLock::new(|| shaders::with_sampling_f32(
                include_str!("../../../shaders/wave_f32.wgsl")
            ));

        // Serialize params matching the WGSL Params struct layout
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.amplitude.to_le_bytes());
        params.extend_from_slice(&self.wavelength.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: WAVE_F32.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: BufferFormat::F32Vec4,
        }])
    }
}
```

### Shader composition helpers

| Helper | Provides | Use For |
|--------|----------|---------|
| `shaders::with_sampling_f32(body)` | `sample_bilinear_f32()` | Distortion filters |
| `shaders::with_io_f32(body)` | `load_pixel()` / `store_pixel()` | Per-pixel ops (liquify) |

### GPU requirements

- WGSL Params struct must have `width: u32, height: u32` as first two fields
- Workgroup size: `[16, 16, 1]` (standard)
- Shaders go in `crates/rasmcore-image/src/shaders/`
- GPU output must match CPU within f32 tolerance

---

## Trait Composition

Filters can implement multiple traits for optimization:

| Trait | Purpose | When to Use |
|-------|---------|-------------|
| `CpuFilter` | **Required.** CPU compute implementation | Always |
| `GpuFilter` | GPU shader dispatch | Embarrassingly parallel operations |
| `PointOp` | 1D LUT for per-channel operations | Brightness, gamma, invert, etc. |
| `ColorOp` | 3D CLUT for color transforms | Hue rotate, saturate, color balance |
| `AnalyticOp` | Expression tree IR for fusion | Simple arithmetic (brightness = input + offset) |

The pipeline fuses consecutive point/color/analytic ops into a single
pass automatically.

---

## Mask Generators and Selective Adjustments

The mask module (`filters/mask/`) provides:

**Generators** (produce grayscale masks):
- `mask_gradient_linear` — Linear gradient with angle/position/feather
- `mask_gradient_radial` — Elliptical gradient from center
- `mask_luminance_range` — Isolate by luminance (highlights/shadows)
- `mask_color_range` — Isolate by hue/saturation
- `mask_from_path` — Rasterize brush stroke points

**Operations**:
- `mask_combine` — Add, subtract, intersect two masks
- `mask_invert` — Invert mask values
- `mask_feather` — Gaussian blur for soft edges

**Compositing**:
- `masked_blend` — `output = adjusted * mask + original * (1 - mask)`

This enables Lightroom-style selective adjustments: apply any filter
stack only within a masked region.

---

## Adding Generators, Compositors, and Mappers

These use V1-style registration macros (not yet migrated to V2):

### Generators

Generators produce images from parameters (no input image):

```rust
#[rasmcore_macros::register_generator(
    name = "checkerboard", category = "generator",
    reference = "alternating two-color grid pattern"
)]
pub fn checkerboard(width: u32, height: u32, cell_size: u32, ...) -> Vec<u8> {
    // Returns RGB8 pixel buffer (width * height * 3 bytes)
}
```

### Compositors (V2)

Compositors combine two images (foreground + background). They implement the
`Compositor` trait and are wrapped by `CompositorNode` with two upstream
connections. Use `#[derive(V2Compositor)]` for automatic registration.

```rust
use rasmcore_pipeline_v2::node::PipelineError;
use rasmcore_pipeline_v2::ops::Compositor;

#[derive(rasmcore_macros::V2Compositor, Clone)]
#[compositor(name = "my_blend", category = "composite", cost = "O(n)")]
pub struct MyBlend {
    /// Blend opacity (0 = bg only, 1 = fully blended)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 1.0)]
    pub opacity: f32,
}

impl Compositor for MyBlend {
    fn compute(
        &self,
        fg: &[f32],
        bg: &[f32],
        _w: u32,
        _h: u32,
    ) -> Result<Vec<f32>, PipelineError> {
        let mut out = Vec::with_capacity(bg.len());
        for (a, b) in fg.chunks_exact(4).zip(bg.chunks_exact(4)) {
            out.push(b[0] * (1.0 - self.opacity) + a[0] * self.opacity);
            out.push(b[1] * (1.0 - self.opacity) + a[1] * self.opacity);
            out.push(b[2] * (1.0 - self.opacity) + a[2] * self.opacity);
            out.push(b[3]); // alpha from background
        }
        Ok(out)
    }
}
```

The `#[derive(V2Compositor)]` macro generates:
- `Default` implementation from `#[param(default = ...)]` values
- `ParamDescriptor` statics for SDK/UI generation
- `CompositorFactoryRegistration` submitted to `inventory`
- `OperationRegistration` with `OperationKind::Compositor`

WIT interface: `apply-compositor(source-a, source-b, name, params)` in `pipeline.wit`.

GPU acceleration: implement `gpu_shader_body()` on your `Compositor` trait impl.
GPU shaders use `load_pixel_a(idx)` and `load_pixel_b(idx)` (dual-input bindings).

Built-in compositors:
- `porter_duff_over` — alpha compositing with opacity control
- `blend_dual` — 25-mode dual-input blend (ISO 32000-2)

### Compositors (V1 — deprecated)

V1 compositors use the `register_compositor` macro (bare function style):

```rust
#[rasmcore_macros::register_compositor(
    name = "blend", category = "composite",
    reference = "27-mode photographic blend"
)]
pub fn blend(
    fg_pixels: &[u8], fg_info: &ImageInfo,
    bg_pixels: &[u8], bg_info: &ImageInfo,
    mode: BlendMode,
) -> Result<Vec<u8>, ImageError> { ... }
```

### Mappers

Mappers change pixel format (e.g., RGB8 to Gray8):

```rust
#[rasmcore_macros::register_mapper(
    name = "grayscale", category = "color",
    reference = "luminance-weighted desaturation",
    output_format = "Gray8"
)]
pub fn grayscale(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    // Returns (new pixels, new ImageInfo with Gray8 format)
}
```

---

## Adding a Decoder

See `crates/rasmcore-image/src/domain/decoder/` for examples. Each format
implements the codec registration pattern with decode functions that produce
`DecodedImage { pixels: Vec<u8>, info: ImageInfo }`.

---

## Adding an Encoder

See `crates/rasmcore-image/src/domain/encoder/` for examples. Encoders
take pixels + ImageInfo + format-specific config and produce encoded bytes.

---

## Pipeline Integration

The V2 pipeline operates in **f32-native mode**:

1. **Ingest**: Source data is promoted to `Rgba32f` immediately after decode
2. **Processing**: All nodes see `Rgba32f` (16 bytes/pixel) — no format dispatch
3. **Output**: Demoted to target format at the encode boundary

GPU dispatch is **primary** — the pipeline tries GPU first, falls back to
CPU. The `GpuFilter` trait enables this automatically.

### Layer Cache

The `LayerCache` persists results across pipeline lifetimes using content
hashes. It supports opt-in quantization:

- `CacheQuality::Full` — 16 bytes/pixel (default)
- `CacheQuality::Q16` — 8 bytes/pixel (2x memory saving)
- `CacheQuality::Q8` — 4 bytes/pixel (4x memory saving)

Quantization is transparent: store() quantizes, get() promotes back to f32.

---

## Generated Files — DO NOT EDIT

The following files are generated by `build.rs` and must not be edited:

- `src/bindings.rs` — WIT component bindings
- `target/*/build/rasmcore-image-*/out/generated_*.rs` — Pipeline nodes, adapters, dispatch
- `target/*/build/rasmcore-image-*/out/param-manifest.json` — Parameter manifest
- `wit/image/filters.wit` — Filter/mapper/compositor WIT declarations (partially generated)

Regenerate with `cargo build -p rasmcore-image` (native) or
`cargo component bindings` (WASM bindings.rs).

---

## Reference Validation

Every filter implementation must cite its reference and include parity tests.
See `docs/REFERENCE_VALIDATION.md` for the full validation standard.

---

## PRNG Requirement

All randomized operations must use deterministic PRNG:

```rust
use rand::SeedableRng;
use rand::rngs::SmallRng;

let mut rng = SmallRng::seed_from_u64(seed);
```

Never use `OsRng`, `thread_rng()`, or any non-deterministic source.
Same seed must produce identical output across platforms.
