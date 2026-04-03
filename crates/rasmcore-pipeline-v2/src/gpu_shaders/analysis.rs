//! GPU shaders for multi-pass analysis filters.
//!
//! These shaders work in multi-dispatch chains:
//! - Pass 1: analysis (histogram reduction, min/max, etc.)
//! - Pass 2+: apply results to pixels
//!
//! Intermediate data (histograms, statistics) is passed between passes
//! via extra_buffers or encoded in the ping-pong output.

/// GPU histogram computation — parallel reduction using workgroup atomics.
///
/// Pass 1: Each workgroup atomically accumulates into a shared 256-bin histogram,
/// then atomically adds to the global histogram in extra_buffers[0].
///
/// Input: f32 pixel data (vec4<f32>)
/// Output: passes through input unchanged (histogram stored in extra buffer)
/// Extra buffer 0 (output): u32[256] histogram bins (luminance)
pub const HISTOGRAM_COMPUTE: &str = r#"
struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height;
  if (idx >= total) { return; }

  let pixel = input[idx];
  // Pass through pixel data unchanged
  output[idx] = pixel;

  // Compute luminance and bin it
  let luma = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
  let bin = u32(clamp(luma * 255.0, 0.0, 255.0));
  atomicAdd(&histogram[bin], 1u);
}
"#;

/// Auto-level LUT application — stretches histogram to full range.
///
/// Pass 2: Reads min/max from extra_buffers, applies stretch per pixel.
/// Params encode the min and max luminance values (computed from histogram on CPU
/// between passes, or via a separate reduction pass).
pub const AUTO_LEVEL_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  min_val: f32,
  max_val: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let pixel = input[idx];
  let range = max(params.max_val - params.min_val, 0.001);
  let inv_range = 1.0 / range;
  output[idx] = vec4<f32>(
    (pixel.x - params.min_val) * inv_range,
    (pixel.y - params.min_val) * inv_range,
    (pixel.z - params.min_val) * inv_range,
    pixel.w,
  );
}
"#;

/// Equalize — apply CDF-based equalization LUT.
///
/// The 256-entry equalization LUT is passed as extra_buffers[0].
/// Each f32 entry maps input luminance bin to output luminance.
pub const EQUALIZE_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> eq_lut: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let pixel = input[idx];
  let luma_in = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
  let bin = u32(clamp(luma_in * 255.0, 0.0, 255.0));
  let luma_out = eq_lut[bin];

  // Scale RGB proportionally to preserve hue
  let scale = select(luma_out / max(luma_in, 0.001), 1.0, luma_in < 0.001);
  output[idx] = vec4<f32>(
    pixel.x * scale,
    pixel.y * scale,
    pixel.z * scale,
    pixel.w,
  );
}
"#;

/// Shadow/Highlight — per-pixel adjustment based on luminance.
///
/// Applied after a luminance blur pass (the blurred luminance field
/// determines which pixels are "shadow" vs "highlight").
/// Blurred luminance is encoded in the alpha channel from a previous pass,
/// or passed as extra_buffers[0].
pub const SHADOW_HIGHLIGHT_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  shadows: f32,
  highlights: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> blurred_luma: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let pixel = input[idx];
  let luma = blurred_luma[idx];

  // Shadow boost: dark areas get brightened
  let shadow_weight = 1.0 - smoothstep(0.0, 0.5, luma);
  let shadow_boost = params.shadows * shadow_weight;

  // Highlight recovery: bright areas get darkened
  let highlight_weight = smoothstep(0.5, 1.0, luma);
  let highlight_reduce = params.highlights * highlight_weight;

  let adjustment = shadow_boost - highlight_reduce;

  output[idx] = vec4<f32>(
    pixel.x + adjustment,
    pixel.y + adjustment,
    pixel.z + adjustment,
    pixel.w,
  );
}
"#;

/// NLM (Non-Local Means) Denoise — per-pixel parallel search.
///
/// Each pixel independently searches its neighborhood for similar patches,
/// computes weights based on patch distance, and accumulates the weighted
/// average. Massively parallel — perfect for GPU despite being compute-heavy.
pub const NLM_DENOISE: &str = r#"
struct Params {
  width: u32,
  height: u32,
  search_radius: u32,
  patch_radius: u32,
  h: f32, // filtering strength
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> vec4<f32> {
  let cx = clamp(x, 0, i32(params.width) - 1);
  let cy = clamp(y, 0, i32(params.height) - 1);
  return input[u32(cx) + u32(cy) * params.width];
}

fn patch_distance(x1: i32, y1: i32, x2: i32, y2: i32) -> f32 {
  var dist = 0.0;
  let r = i32(params.patch_radius);
  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let p1 = get_pixel(x1 + dx, y1 + dy);
      let p2 = get_pixel(x2 + dx, y2 + dy);
      let d = p1 - p2;
      dist += dot(d.xyz, d.xyz);
    }
  }
  let patch_size = f32((2 * r + 1) * (2 * r + 1));
  return dist / (patch_size * 3.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  let sr = i32(params.search_radius);
  let h2 = params.h * params.h;

  var sum = vec4<f32>(0.0);
  var weight_sum = 0.0;

  for (var sy = -sr; sy <= sr; sy++) {
    for (var sx = -sr; sx <= sr; sx++) {
      let dist = patch_distance(x, y, x + sx, y + sy);
      let weight = exp(-dist / h2);
      sum += weight * get_pixel(x + sx, y + sy);
      weight_sum += weight;
    }
  }

  let idx = gid.x + gid.y * params.width;
  let inv_w = select(1.0 / weight_sum, 0.0, weight_sum < 1e-10);
  output[idx] = vec4<f32>(sum.xyz * inv_w, input[idx].w);
}
"#;

/// Dehaze — dark channel prior estimation + atmosphere removal.
///
/// Pass 1: compute dark channel (min RGB in local patch)
/// This is a reduction-like operation per pixel.
pub const DEHAZE_DARK_CHANNEL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  patch_radius: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  let r = i32(params.patch_radius);
  var dark_min = 1.0e30;

  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let sx = clamp(x + dx, 0, i32(params.width) - 1);
      let sy = clamp(y + dy, 0, i32(params.height) - 1);
      let p = input[u32(sx) + u32(sy) * params.width];
      dark_min = min(dark_min, min(p.x, min(p.y, p.z)));
    }
  }

  // Store dark channel in output (will be used in pass 2)
  let idx = gid.x + gid.y * params.width;
  let pixel = input[idx];
  output[idx] = vec4<f32>(pixel.xyz, dark_min);
}
"#;

/// Dehaze — atmosphere removal using dark channel.
///
/// Pass 2: uses dark channel (stored in alpha from pass 1) to remove haze.
pub const DEHAZE_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  atmosphere_r: f32,
  atmosphere_g: f32,
  atmosphere_b: f32,
  strength: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let pixel = input[idx];
  let dark_channel = pixel.w; // from pass 1

  // Transmission estimate
  let t = max(1.0 - params.strength * dark_channel, 0.1);
  let inv_t = 1.0 / t;

  // Remove atmosphere
  let atm = vec3<f32>(params.atmosphere_r, params.atmosphere_g, params.atmosphere_b);
  let dehazed = (pixel.xyz - atm) * inv_t + atm;

  output[idx] = vec4<f32>(dehazed, 1.0);
}
"#;

/// Retinex SSR — single-scale retinex (log domain).
///
/// Applied after a Gaussian blur pass: retinex = log(input) - log(blur(input))
/// The blurred image comes from a previous blur dispatch.
pub const RETINEX_SSR_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  gain: f32,
  offset: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> blurred: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let pixel = input[idx];
  let blur_pixel = blurred[idx];

  // Retinex: log(input) - log(blur)
  let log_input = vec3<f32>(
    log(max(pixel.x, 1e-6)),
    log(max(pixel.y, 1e-6)),
    log(max(pixel.z, 1e-6)),
  );
  let log_blur = vec3<f32>(
    log(max(blur_pixel.x, 1e-6)),
    log(max(blur_pixel.y, 1e-6)),
    log(max(blur_pixel.z, 1e-6)),
  );

  let retinex = (log_input - log_blur) * params.gain + params.offset;

  output[idx] = vec4<f32>(retinex, pixel.w);
}
"#;

/// Dither (ordered) — blue noise threshold per pixel.
///
/// Uses a Bayer matrix pattern for ordered dithering.
pub const DITHER_ORDERED: &str = r#"
struct Params {
  width: u32,
  height: u32,
  levels: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

// 8x8 Bayer matrix (normalized to [0,1])
fn bayer8(x: u32, y: u32) -> f32 {
  let bx = x % 8u;
  let by = y % 8u;
  // Compute Bayer value via bit interleaving
  var value = 0u;
  for (var bit = 0u; bit < 3u; bit++) {
    let mask = 1u << bit;
    value |= ((bx & mask) << (bit + 1u)) | ((by & mask) << bit);
  }
  return f32(value) / 64.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.x + gid.y * params.width;

  let pixel = input[idx];
  let threshold = bayer8(gid.x, gid.y);
  let inv_levels = 1.0 / f32(params.levels - 1u);

  let r = floor(pixel.x * f32(params.levels - 1u) + threshold) * inv_levels;
  let g = floor(pixel.y * f32(params.levels - 1u) + threshold) * inv_levels;
  let b = floor(pixel.z * f32(params.levels - 1u) + threshold) * inv_levels;

  output[idx] = vec4<f32>(r, g, b, pixel.w);
}
"#;

/// Normalize — stretch to [0,1] range.
///
/// Params encode precomputed min/max per channel.
pub const NORMALIZE_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  min_r: f32,
  max_r: f32,
  min_g: f32,
  max_g: f32,
  min_b: f32,
  max_b: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let pixel = input[idx];
  output[idx] = vec4<f32>(
    (pixel.x - params.min_r) / max(params.max_r - params.min_r, 0.001),
    (pixel.y - params.min_g) / max(params.max_g - params.min_g, 0.001),
    (pixel.z - params.min_b) / max(params.max_b - params.min_b, 0.001),
    pixel.w,
  );
}
"#;
