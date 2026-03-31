# Extending rasmcore

This guide explains how to add custom filters, codecs, transforms, and
other operations to rasmcore. It covers both extending the core crate
and building external plugin crates.

---

## Table of Contents

1. [Extension Model Overview](#extension-model-overview)
2. [Adding a Filter](#adding-a-filter)
3. [ConfigParams — Typed Parameter Structs](#configparams--typed-parameter-structs)
4. [Point Operations vs Spatial Operations](#point-operations-vs-spatial-operations) (includes PointOp trait for LUT fusion)
5. [Adding a Decoder](#adding-a-decoder)
6. [Adding an Encoder](#adding-an-encoder)
7. [Adding Generators, Compositors, and Mappers](#adding-generators-compositors-and-mappers)
8. [Pipeline Integration](#pipeline-integration)
9. [Generated Files — DO NOT EDIT](#generated-files--do-not-edit)
10. [External Crates / Plugin Model](#external-crates--plugin-model)
11. [Reference Validation](#reference-validation)
12. [PRNG Requirement](#prng-requirement)

---

## Extension Model Overview

rasmcore uses a **registration macro + codegen** architecture:

1. You write a function and annotate it with a registration macro
   (e.g., `#[register_filter]`)
2. The `rasmcore-macros` crate generates an `inventory` registration at
   compile time
3. The `rasmcore-codegen` crate (invoked by `build.rs`) parses your source,
   extracts registrations, and generates pipeline nodes, SDK methods,
   CLI dispatch, WIT declarations, and a parameter manifest
4. You never edit generated files — you edit source and rebuild

```
Source Code                    Codegen (build.rs)              Generated Output
─────────────                  ──────────────────              ────────────────
#[register_filter]    ──parse──▶  CodegenData    ──generate──▶  pipeline nodes
#[derive(ConfigParams)]                                         SDK methods
                                                                CLI dispatch
                                                                WIT declarations
                                                                param manifest
```

---

## Adding a Filter

### Step 1: Define a ConfigParams struct

Every parameterized filter must have a ConfigParams struct. The struct name
must be `{PascalCaseFilterName}Params` — codegen links them by naming
convention.

```rust
// In crates/rasmcore-image/src/domain/filters.rs

/// Parameters for median filter.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct MedianParams {
    /// Filter radius in pixels
    #[param(min = 1, max = 20, step = 1, default = 3, hint = "rc.log_slider")]
    pub radius: u32,
}
```

Zero-parameter filters (e.g., `invert`, `grayscale`) do not need a
ConfigParams struct.

### Step 2: Write the filter function

The function signature must be:

```rust
pub fn my_filter(
    pixels: &[u8],
    info: &ImageInfo,
    config: &MyFilterParams,  // omit for zero-param filters
) -> Result<Vec<u8>, ImageError>
```

- `pixels` — Input pixel buffer (packed, row-major)
- `info` — Image dimensions, pixel format, color space
- `config` — Reference to your ConfigParams struct
- Returns a new pixel buffer of the same length, or an error

### Step 3: Register with `#[register_filter]`

```rust
#[rasmcore_macros::register_filter(
    name = "median",
    category = "spatial",
    group = "denoise",
    variant = "median",
    reference = "median rank filter"
)]
pub fn median(
    pixels: &[u8],
    info: &ImageInfo,
    config: &MedianParams,
) -> Result<Vec<u8>, ImageError> {
    let radius = config.radius;
    // ... implementation ...
}
```

#### Registration attributes

| Attribute   | Required | Description |
|-------------|----------|-------------|
| `name`      | Yes      | Unique identifier used in WIT, SDK, and CLI |
| `category`  | No       | Grouping: `spatial`, `color`, `adjustment`, `edge`, `morphology`, `effect`, `enhancement`, `distortion`, `transform`, `generator`, `composite`, `alpha`, `grading`, `tonemapping`, `threshold`, `analysis`, `tool`, `draw`, `advanced` |
| `group`     | No       | Sub-group for UI (e.g., `blur`, `denoise`, `edge_detect`) |
| `variant`   | No       | Variant name within the group |
| `reference` | No       | Algorithm attribution (e.g., `"Canny 1986"`, `"OpenCV cv2.medianBlur"`) |
| `overlap`   | No       | Tile overlap for spatial ops: `"zero"` (default), `"uniform(N)"`, `"param(name)"`, `"full"` |
| `output_format` | No   | Output pixel format override (used by mapper pipeline nodes) |

### Step 4: Add tests

Every filter must include:

1. **Basic functionality test** — Verify output is correct for known input
2. **Format coverage** — Test with Rgb8, Rgba8, Gray8 at minimum
3. **Edge cases** — Zero-size, single pixel, parameter extremes
4. **Reference parity test** — Compare against a reference implementation
   (see [Reference Validation](#reference-validation))

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_basic() {
        let pixels = vec![0, 128, 255, 64, 192, 32, 96, 160, 224];
        let info = ImageInfo {
            width: 3, height: 3,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let result = median(&pixels, &info, &MedianParams { radius: 1 }).unwrap();
        assert_eq!(result.len(), pixels.len());
    }
}
```

### Step 5: Rebuild

```bash
cargo build -p rasmcore-image
```

Codegen automatically runs via `build.rs` and produces pipeline nodes,
SDK methods, CLI dispatch, and manifest entries for your filter.

### Complete example: zero-parameter filter

```rust
#[rasmcore_macros::register_filter(
    name = "invert",
    category = "adjustment",
    reference = "channel value inversion"
)]
pub fn invert(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    Ok(pixels.iter().map(|&v| 255 - v).collect())
}
```

No ConfigParams struct needed. Codegen detects zero parameters and
generates the appropriate signatures.

---

## ConfigParams — Typed Parameter Structs

### The `#[derive(ConfigParams)]` macro

This derive macro generates three things:

1. `param_descriptors()` — Returns parameter metadata for the manifest
2. `config_hint()` — Returns a UI hint string for the whole struct
3. `Default` impl — Uses `#[param(default = ...)]` values

### Field attributes

```rust
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct BlurParams {
    /// Blur radius in pixels          // ← doc comment becomes the label
    #[param(
        min = 0.0,                     // minimum value
        max = 100.0,                   // maximum value
        step = 0.5,                    // UI slider step
        default = 3.0,                 // default value
        hint = "rc.log_slider"         // UI hint (optional)
    )]
    pub radius: f32,
}
```

### Struct-level hint

```rust
#[derive(rasmcore_macros::ConfigParams, Clone)]
#[config_hint("rc.color_rgba")]        // UI hint for the whole struct
pub struct ColorRgba {
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub r: u8,
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub g: u8,
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub b: u8,
    #[param(min = 0, max = 255, step = 1, default = 255)]
    pub a: u8,
}
```

### Nested ConfigParams

Structs can embed other ConfigParams structs. Fields are auto-flattened
with dot-notation in the manifest (e.g., `color.r`, `color.g`):

```rust
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct VignetteParams {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub strength: f32,
    pub color: ColorRgba,   // flattened to color.r, color.g, color.b, color.a
}
```

### UI hint vocabulary

See `.agent/kf/code_styleguides/param-hints.md` for the full hint
vocabulary: `rc.angle_deg`, `rc.percentage`, `rc.opacity`, `rc.pixels`,
`rc.log_slider`, `rc.seed`, `rc.temperature_k`, etc.

---

## Point Operations vs Spatial Operations

### Point operations

Point operations (brightness, contrast, invert, color adjustments) process
each pixel independently. They need no overlap — the output pixel depends
only on the input pixel at the same position.

Point operations use the default `input_rect()` (returns output rect
unchanged). No `overlap` attribute needed.

#### Implementing point ops with the PointOp trait (LUT fusion)

If your filter is a **per-channel mapping** (each output channel value
depends only on the corresponding input channel value), implement the
`PointOp` trait on your ConfigParams struct. This enables automatic
**LUT fusion** in the pipeline — consecutive point ops are composed into
a single 256-entry lookup table and applied in one memory pass.

```rust
use crate::domain::point_ops::{PointOp, apply_lut};

#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct BrightnessParams {
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0)]
    pub amount: f32,
}

impl PointOp for BrightnessParams {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        let offset = (self.amount * 255.0).round() as i16;
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = (i as i16 + offset).clamp(0, 255) as u8;
        }
        lut
    }
}

#[register_filter(name = "brightness", category = "adjustment")]
pub fn brightness(pixels: &[u8], info: &ImageInfo, config: &BrightnessParams)
    -> Result<Vec<u8>, ImageError>
{
    apply_lut(pixels, info, &config.build_lut())
}
```

The filter function body is always the same one-liner:
`apply_lut(pixels, info, &config.build_lut())`. The `PointOp` trait is
what enables the pipeline optimizer to fuse your filter with adjacent
point ops automatically.

#### How LUT fusion works

When the pipeline encounters consecutive point-op nodes, it composes
their LUTs at plan time and replaces them with a single fused node:

```
Pipeline DAG:
  source -> brightness -> contrast -> gamma -> blur -> levels -> invert -> write

Fusion (plan time):
  Run 1: [brightness, contrast, gamma]  -> compose into 1 LUT (O(256))
  Barrier: blur (spatial op, breaks fusion)
  Run 2: [levels, invert]              -> compose into 1 LUT (O(256))

Execution:
  source -> fused_lut_1 -> blur -> fused_lut_2 -> write
  (3 adjustments = 1 memory pass)    (2 adjustments = 1 memory pass)
```

LUT composition is `fused[i] = second[first[i]]` — 256 iterations,
trivial cost. The savings come from eliminating memory passes: on a
100MP RGBA8 image, each pass touches 400MB. Fusing 5 ops saves 1.6GB
of memory bandwidth.

LUT fusion is a plan-time optimization — all LUTs are composed before
any pixels are processed. At runtime, each fused chain is a single
`apply_lut` pass: one table lookup per channel per pixel, no math.

#### When NOT to use PointOp

Do **not** implement `PointOp` if your filter:

- Depends on **multiple channels** (e.g., hue rotation needs R, G, and B
  together to compute each output channel)
- Depends on **pixel position** (e.g., vignette, gradient map)
- Depends on **neighboring pixels** (any spatial operation)

Multi-channel per-pixel operations (hue, saturation, color grading) are
a different category and will use 3D CLUT fusion in the future.

### Spatial operations

Spatial operations (blur, sharpen, median, morphology, edge detection)
read a neighborhood around each pixel. When the pipeline processes tiles,
spatial operations need extra pixels at tile boundaries to compute correct
output — this is called **overlap**.

Codegen automatically detects overlap from your ConfigParams fields:

| Field name    | Expansion formula          |
|---------------|----------------------------|
| `radius`      | expand by `radius` pixels  |
| `blur_radius` | expand by `blur_radius`    |
| `ksize`       | expand by `ksize / 2`      |
| `sigma`       | expand by `ceil(3 * sigma)` |
| `diameter`    | expand by `diameter / 2`   |
| `search_size` | expand by `search_size / 2` |
| `length`      | expand by `length`         |

For filters with fixed kernels (no configurable size), use the `overlap`
registration attribute:

```rust
#[rasmcore_macros::register_filter(
    name = "canny",
    category = "edge",
    overlap = "uniform(2)"    // fixed 2px overlap for 3x3 Sobel + NMS
)]
```

#### Overlap attribute values

| Value           | Meaning |
|-----------------|---------|
| `"zero"`        | No overlap (default, point operation) |
| `"uniform(N)"`  | Fixed N pixels on all sides |
| `"param(name)"` | Dynamic — uses the runtime value of the named parameter |
| `"full"`        | Requests the full image (non-tileable operation) |

### How input_rect works

Each pipeline node implements `input_rect()` on the `ImageNode` trait.
Given the output rect the downstream node needs, `input_rect()` returns
the (larger) input rect this node needs from upstream. Codegen generates
this automatically based on your ConfigParams fields or overlap attribute.

```
Output tile requested: (10, 10, 32, 32)
Blur with radius=3:    input_rect → (7, 7, 38, 38)  // expanded by 3 on each side
Pipeline fetches the larger region, runs the filter, crops to (10, 10, 32, 32)
```

### Affine Transform Composition

Transform nodes (resize, crop, rotate, flip) implement `AffineOp` to enable
single-resample optimization. When the pipeline chains multiple transforms,
the optimizer composes their affine matrices into one and resamples once —
eliminating multi-pass interpolation artifacts.

#### The `AffineOp` trait

```rust
pub trait AffineOp {
    /// Return the 2x3 affine matrix and output dimensions for this transform.
    fn to_affine(&self) -> ([f64; 6], u32, u32);
}
```

The matrix format is `[a, b, tx, c, d, ty]` representing:
- `x' = a*x + b*y + tx`
- `y' = c*x + d*y + ty`

#### Implementing AffineOp for a new transform

If your transform node is expressible as a 2x3 affine matrix, implement
`AffineOp` and override `as_affine_op()` in `ImageNode`:

```rust
impl AffineOp for MyTransformNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        // Return (matrix, output_width, output_height)
        ([1.0, 0.0, -self.x as f64, 0.0, 1.0, -self.y as f64],
         self.width, self.height)
    }
}

impl ImageNode for MyTransformNode {
    // ... other methods ...
    fn as_affine_op(&self) -> Option<([f64; 6], u32, u32)> {
        Some(self.to_affine())
    }
    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }
}
```

Both `as_affine_op()` and `upstream_id()` must be implemented for the
optimizer to walk the chain. Non-affine nodes (blur, filters) act as
composition barriers — the optimizer stops at them.

#### Matrix composition

Use `compose_affine(outer, inner)` to multiply two 2x3 matrices. The
result applies `inner` first, then `outer`. The optimizer calls this
automatically when fusing consecutive affine nodes.

#### When NOT to implement AffineOp

- **Nonlinear transforms** (barrel distortion, swirl, perspective warp)
  — these are not affine and cannot be composed with a 2x3 matrix
- **Format-changing operations** (grayscale, flatten) — these change the
  pixel format, not spatial layout
- **Content-dependent transforms** (seam carving, smart crop) — output
  depends on image content, not just geometry

---

## Adding a Decoder

### Registration macro

```rust
#[rasmcore_macros::register_decoder(
    name = "PNG Decoder",
    formats = "png"           // space or comma-separated format identifiers
)]
pub fn decode_png(data: &[u8]) -> Result<DecodedImage, ImageError> {
    // ... implementation ...
    Ok(DecodedImage { pixels, info })
}
```

### Registration struct

The macro generates a `StaticDecoderRegistration`:

```rust
pub struct StaticDecoderRegistration {
    pub name: &'static str,
    pub formats: &'static str,
    pub fn_name: &'static str,
}
```

Registrations are collected via `inventory` and available at runtime through
`registered_decoders()`.

### Decoder function requirements

- Input: raw encoded bytes (`&[u8]`)
- Output: `Result<DecodedImage, ImageError>` containing decoded pixels + metadata
- Must detect and handle format variants (bit depth, color mode, compression)
- Must validate against reference decoders (see `codec-validation.md`)

---

## Adding an Encoder

### Registration macro

```rust
#[rasmcore_macros::register_encoder(
    name = "PNG Encoder",
    format = "png",
    mime = "image/png",
    extensions = "png"        // comma-separated file extensions
)]
pub fn encode_png(
    pixels: &[u8],
    info: &ImageInfo,
    config: &PngEncodeConfig,
) -> Result<Vec<u8>, ImageError> {
    // ... implementation ...
}
```

### Registration struct

```rust
pub struct StaticEncoderRegistration {
    pub name: &'static str,
    pub format: &'static str,
    pub mime: &'static str,
    pub extensions: &'static [&'static str],
    pub fn_name: &'static str,
}
```

### Encoder function requirements

- Input: raw pixels + ImageInfo + format-specific config
- Output: `Result<Vec<u8>, ImageError>` containing encoded bytes
- Must validate roundtrip: encode → decode → compare
- Must validate against reference encoders (see `encoder-params.md`)

---

## Adding Generators, Compositors, and Mappers

### Generators — procedural image sources

Generators create images from parameters (no pixel input):

```rust
#[rasmcore_macros::register_generator(
    name = "perlin_noise",
    category = "noise"
)]
pub fn perlin_noise(width: u32, height: u32, seed: u32) -> Vec<u8> {
    // Generate image from parameters — no input pixels
}
```

### Compositors — multi-input blending

Compositors combine two or more images:

```rust
#[rasmcore_macros::register_compositor(
    name = "blend_normal",
    category = "composite",
    group = "blend",
    variant = "normal"
)]
pub fn blend_normal(
    pixels_a: &[u8], info_a: &ImageInfo,
    pixels_b: &[u8], info_b: &ImageInfo,
    opacity: f32,
) -> Vec<u8> {
    // Blend two images
}
```

### Mappers — format-changing operations

Mappers transform pixel format (e.g., RGB8 to Gray8):

```rust
#[rasmcore_macros::register_mapper(
    name = "to_grayscale",
    category = "color"
)]
pub fn to_grayscale(pixels: &[u8], info: &ImageInfo) -> Vec<u8> {
    // Convert pixel format — output dimensions match input,
    // but pixel format may change
}
```

---

## Pipeline Integration

### How filters become pipeline nodes

When you register a filter, codegen generates a pipeline node struct and
an `ImageNode` trait implementation. You do not write this code — it is
generated automatically.

For a filter with signature `fn blur(pixels, info, config: &BlurParams)`,
codegen produces:

```rust
// AUTO-GENERATED — do not edit
pub struct BlurNode {
    upstream: u32,
    source_info: ImageInfo,
    config: BlurParams,
}

impl ImageNode for BlurNode {
    fn info(&self) -> ImageInfo { self.source_info.clone() }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let upstream_rect = self.input_rect(request, self.source_info.width, self.source_info.height);
        let src_pixels = upstream_fn(self.upstream, upstream_rect)?;
        // ... calls filters::blur(&src_pixels, &region_info, &self.config) ...
        // ... crops result back to requested region ...
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        output.expand_uniform(self.config.radius as u32, bounds_w, bounds_h)
    }
}
```

### The ImageNode trait

```rust
pub trait ImageNode {
    /// Image dimensions and format.
    fn info(&self) -> ImageInfo;

    /// Compute pixels for the requested region.
    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError>;

    /// Input rect needed to produce the given output rect.
    /// Default: no expansion (point operation).
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        output.clamp(bounds_w, bounds_h)
    }

    /// Access pattern hint for cache optimization.
    fn access_pattern(&self) -> AccessPattern;
}
```

### Manual pipeline nodes (transforms)

Transform nodes (resize, crop, rotate, flip) are **not** generated by
codegen — they are hand-written because they change output dimensions or
need custom input_rect logic. They follow the same `ImageNode` trait:

```rust
pub struct ResizeNode {
    upstream: u32,
    target_width: u32,
    target_height: u32,
    filter: ResizeFilter,
    source_info: ImageInfo,
}

impl ImageNode for ResizeNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.target_width,
            height: self.target_height,
            ..self.source_info.clone()
        }
    }

    fn compute_region(&self, request: Rect, upstream_fn: ...) -> Result<Vec<u8>, ImageError> {
        // Map output coordinates back to source coordinates
        // Request the corresponding source region
        // Perform the resize
    }
}
```

---

## Generated Files — DO NOT EDIT

The following files are produced by codegen during `cargo build`. They
live in `target/*/build/rasmcore-image-*/out/`. **Never edit these files
directly** — your changes will be overwritten on the next build.

| Generated File | Source Generator | Purpose |
|----------------|-----------------|---------|
| `generated_filter_adapter.rs` | `adapter::generate()` | WIT guest dispatch for filters |
| `generated_pipeline_nodes.rs` | `pipeline::generate_nodes()` | Pipeline node structs + ImageNode impls |
| `generated_pipeline_adapter.rs` | `pipeline::generate_adapter_macro()` | Pipeline filter method macro |
| `generated_pipeline_mapper_adapter.rs` | `pipeline_mapper::generate_mapper_adapter_macro()` | Pipeline mapper method macro |
| `generated_sdk_rust.rs` | `sdk_rust::generate()` | Rust native SDK (RcImage builder) |
| `generated_cli_dispatch.rs` | `cli_dispatch::generate()` | CLI command dispatch table |
| `param-manifest.json` | `manifest::generate()` | Filter/parameter catalog (JSON) |
| `param-manifest.hash` | `fnv1a_64()` | Manifest version hash |

WIT declarations are printed to stderr during build for manual review.

### How to regenerate

```bash
cargo build -p rasmcore-image   # triggers build.rs → codegen
```

### How to add a new generated output

If you are a maintainer adding a new codegen output:

1. Add your generator function in `crates/rasmcore-codegen/src/generate/`
2. Call it from `generate_all()` in `generate/mod.rs`
3. Write the output to `out_dir.join("your_file.rs")`
4. Include it in the appropriate source file with `include!(concat!(env!("OUT_DIR"), "/your_file.rs"))`

---

## External Crates / Plugin Model

rasmcore's registration system uses the `inventory` crate, which collects
registrations across crate boundaries at link time. This means external
crates can register their own filters, codecs, and operations.

### Using registration macros from external crates

```toml
# In your Cargo.toml
[dependencies]
rasmcore-image = { path = "../rasmcore-image" }  # or version
rasmcore-macros = { path = "../rasmcore-macros" }
inventory = "0.3"
```

```rust
// In your crate's lib.rs
use rasmcore_image::domain::types::*;
use rasmcore_image::domain::error::ImageError;

#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct MyCustomBlurParams {
    #[param(min = 0.0, max = 50.0, step = 0.5, default = 5.0)]
    pub radius: f32,
}

#[rasmcore_macros::register_filter(
    name = "my_custom_blur",
    category = "spatial",
    reference = "My custom algorithm"
)]
pub fn my_custom_blur(
    pixels: &[u8],
    info: &ImageInfo,
    config: &MyCustomBlurParams,
) -> Result<Vec<u8>, ImageError> {
    // Your implementation
    todo!()
}
```

The `inventory` crate ensures your registration is collected when your
crate is linked into the final binary. The `registered_filters()` function
will include your filter alongside the built-in ones.

### Manual pipeline node implementation

For filters in external crates, codegen does not automatically generate
pipeline nodes (it only parses the main `filters.rs` file). You can
implement `ImageNode` manually:

```rust
use rasmcore_image::domain::pipeline::graph::{ImageNode, AccessPattern};
use rasmcore_pipeline::Rect;

pub struct MyCustomBlurNode {
    upstream: u32,
    source_info: ImageInfo,
    config: MyCustomBlurParams,
}

impl ImageNode for MyCustomBlurNode {
    fn info(&self) -> ImageInfo { self.source_info.clone() }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let upstream_rect = self.input_rect(
            request, self.source_info.width, self.source_info.height
        );
        let src_pixels = upstream_fn(self.upstream, upstream_rect)?;
        let region_info = ImageInfo {
            width: upstream_rect.width,
            height: upstream_rect.height,
            ..self.source_info
        };
        let filtered = my_custom_blur(&src_pixels, &region_info, &self.config)?;

        // Crop back to requested region if expanded
        if upstream_rect == request {
            Ok(filtered)
        } else {
            let bpp = bytes_per_pixel(self.source_info.format);
            let sub = Rect::new(
                request.x - upstream_rect.x,
                request.y - upstream_rect.y,
                request.width,
                request.height,
            );
            Ok(crop_region(&filtered, Rect::new(0, 0, upstream_rect.width, upstream_rect.height), sub, bpp))
        }
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        output.expand_uniform(self.config.radius as u32, bounds_w, bounds_h)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::LocalNeighborhood
    }
}
```

---

## Reference Validation

**Every implementation must be validated against an authoritative reference.**
This is not optional. See `docs/REFERENCE_VALIDATION.md` for the full
validation principle and existing parity records.

### Requirements

1. **Cite the reference** — Document which implementation you are matching
   (OpenCV, ImageMagick, GEGL, libvips, a specific paper, etc.) in the
   `reference` attribute of your registration macro.

2. **Write a parity test** — Compare your output against the reference
   for at least one non-trivial input. Document the alignment metric:
   - **Bit-exact** — Output is byte-identical to reference
   - **MAE < N** — Mean absolute error per pixel below threshold
   - **PSNR > N dB** — Peak signal-to-noise ratio above threshold

3. **Document residuals** — If your output differs from the reference,
   document exactly why (e.g., floating-point precision, different
   boundary handling, intentional algorithm improvement).

### Example parity test

```rust
#[test]
fn median_parity_vs_opencv() {
    // Generate test image
    let (pixels, info) = make_gradient(64, 64, PixelFormat::Gray8);

    // Run our implementation
    let ours = median(&pixels, &info, &MedianParams { radius: 2 }).unwrap();

    // Run OpenCV via Python subprocess
    let reference = run_opencv_median(&pixels, &info, 2);

    // Compare
    let mae = mean_absolute_error(&ours, &reference);
    assert!(mae < 1.0, "Median MAE vs OpenCV: {mae} (expected < 1.0)");
}
```

### Codec validation

Codecs follow a stricter **three-way validation** standard. See
`.agent/kf/code_styleguides/codec-validation.md` for details.

---

## PRNG Requirement

Any algorithm that involves randomness **must use a seeded PRNG** — never
`OsRng`, `thread_rng()`, or any non-deterministic source.

### Why

- Tests must be deterministic and reproducible
- Same input + same seed = same output, always
- Cross-platform reproducibility (WASM, native, CI)

### How

```rust
use rand::rngs::SmallRng;
use rand::SeedableRng;

pub fn film_grain(
    pixels: &[u8],
    info: &ImageInfo,
    config: &FilmGrainParams,
) -> Result<Vec<u8>, ImageError> {
    let mut rng = SmallRng::seed_from_u64(config.seed);
    // Use rng for all random operations
}
```

The `seed` parameter should be exposed in the ConfigParams struct so
users can control reproducibility:

```rust
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct FilmGrainParams {
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 25.0)]
    pub amount: f32,
    #[param(min = 0, max = 999999, step = 1, default = 42, hint = "rc.seed")]
    pub seed: u64,
}
```
