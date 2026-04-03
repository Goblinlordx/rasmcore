//! GPU shaders for enhancement filters that need multi-pass or custom dispatch.
//!
//! Some enhancement filters (Clarity, FrequencyHigh, ShadowHighlight, Retinex,
//! PyramidDetailRemap, Clahe) require blur passes + apply shaders. The blur
//! pass reuses spatial::GAUSSIAN_BLUR_H/V; only the apply shaders live here.

/// Clarity apply — midtone-weighted local contrast blend.
///
/// Dispatched after a blur pass. extra_buffers[0] = original image.
/// `output = original + amount * (original - blurred) * midtone_weight(luma)`
pub const CLARITY_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  amount: f32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> blurred: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> original: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let orig = original[idx];
  let blur = blurred[idx];
  let luma = clamp(0.2126 * orig.x + 0.7152 * orig.y + 0.0722 * orig.z, 0.0, 1.0);
  let weight = 4.0 * luma * (1.0 - luma); // midtone peak at 0.5

  output[idx] = vec4<f32>(
    orig.x + params.amount * (orig.x - blur.x) * weight,
    orig.y + params.amount * (orig.y - blur.y) * weight,
    orig.z + params.amount * (orig.z - blur.z) * weight,
    orig.w,
  );
}
"#;

/// Frequency high-pass apply — subtract blur, add 0.5 offset.
///
/// Dispatched after blur pass. extra_buffers[0] = original image.
pub const FREQUENCY_HIGH_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read> blurred: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> original: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let orig = original[idx];
  let blur = blurred[idx];
  output[idx] = vec4<f32>(
    (orig.x - blur.x) + 0.5,
    (orig.y - blur.y) + 0.5,
    (orig.z - blur.z) + 0.5,
    orig.w,
  );
}
"#;

/// Shadow/Highlight apply — luminance-based local tone mapping.
///
/// Dispatched after blur-luma pass. extra_buffers[0] = blurred luminance (f32 per pixel).
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

  let shadow_w = 1.0 - smoothstep(0.0, 0.5, luma);
  let highlight_w = smoothstep(0.5, 1.0, luma);
  let adj = params.shadows * shadow_w - params.highlights * highlight_w;

  output[idx] = vec4<f32>(
    pixel.x + adj,
    pixel.y + adj,
    pixel.z + adj,
    pixel.w,
  );
}
"#;

/// Extract luminance to single f32 buffer (for ShadowHighlight blur-luma pass).
pub const EXTRACT_LUMA: &str = r#"
struct Params {
  width: u32,
  height: u32,
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
  let luma = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
  // Store original RGB with luma in alpha for downstream
  output[idx] = vec4<f32>(pixel.xyz, luma);
}
"#;

/// Retinex SSR apply — log-domain illumination removal.
///
/// Dispatched after blur pass. Input is blurred; extra_buffers[0] = original.
pub const RETINEX_SSR_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  gain: f32,
  offset: f32,
}
@group(0) @binding(0) var<storage, read> blurred: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> original: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let orig = original[idx];
  let blur = blurred[idx];

  let log_orig = vec3<f32>(
    log(max(orig.x, 1e-6)),
    log(max(orig.y, 1e-6)),
    log(max(orig.z, 1e-6)),
  );
  let log_blur = vec3<f32>(
    log(max(blur.x, 1e-6)),
    log(max(blur.y, 1e-6)),
    log(max(blur.z, 1e-6)),
  );

  let retinex = (log_orig - log_blur) * params.gain + params.offset;
  output[idx] = vec4<f32>(retinex, orig.w);
}
"#;

/// Retinex MSR accumulate — accumulate log-ratio from one blur scale.
///
/// Dispatched after each blur pass. extra_buffers[0] = original.
/// Reduction buffer = accumulator (read_write, accumulated across scales).
pub const RETINEX_MSR_ACCUMULATE: &str = r#"
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read> blurred: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> original: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> accumulator: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let orig = original[idx];
  let blur = blurred[idx];

  let log_orig = vec3<f32>(
    log(max(orig.x, 1e-6)),
    log(max(orig.y, 1e-6)),
    log(max(orig.z, 1e-6)),
  );
  let log_blur = vec3<f32>(
    log(max(blur.x, 1e-6)),
    log(max(blur.y, 1e-6)),
    log(max(blur.z, 1e-6)),
  );

  let prev = accumulator[idx];
  accumulator[idx] = vec4<f32>(prev.xyz + (log_orig - log_blur), 0.0);
  // Pass through original for next blur pass
  output[idx] = orig;
}
"#;

/// Retinex MSR normalize — average accumulated scales and normalize to [0,1].
pub const RETINEX_MSR_NORMALIZE: &str = r#"
struct Params {
  width: u32,
  height: u32,
  num_scales: f32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> accumulator: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> minmax: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let acc = accumulator[idx].xyz / params.num_scales;
  let ch_min = minmax[0].xyz;
  let ch_max = minmax[1].xyz;
  let range = max(ch_max - ch_min, vec3<f32>(1e-10));

  output[idx] = vec4<f32>(
    (acc.x - ch_min.x) / range.x,
    (acc.y - ch_min.y) / range.y,
    (acc.z - ch_min.z) / range.z,
    input[idx].w,
  );
}
"#;

/// CLAHE — tiled adaptive histogram equalization.
///
/// Single-pass: each pixel determines its tile, looks up the pre-computed
/// CLAHE LUT (passed via extra_buffer), bilinearly interpolates between tiles.
///
/// extra_buffers[0]: f32 LUT array [grid_x * grid_y * 256] — one 256-entry LUT per tile.
pub const CLAHE_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  grid: u32,
  tile_w: u32,
  tile_h: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> tile_luts: array<f32>;

fn lut_lookup(tile_idx: u32, bin: u32) -> f32 {
  return tile_luts[tile_idx * 256u + bin];
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = idx % params.width;
  let y = idx / params.width;

  let pixel = input[idx];
  let luma = clamp(0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z, 0.0, 1.0);
  let bin = u32(luma * 255.0);

  let tx_f = clamp(f32(x) / f32(params.tile_w) - 0.5, 0.0, f32(params.grid - 1u));
  let ty_f = clamp(f32(y) / f32(params.tile_h) - 0.5, 0.0, f32(params.grid - 1u));
  let tx0 = u32(tx_f);
  let ty0 = u32(ty_f);
  let tx1 = min(tx0 + 1u, params.grid - 1u);
  let ty1 = min(ty0 + 1u, params.grid - 1u);
  let fx = tx_f - f32(tx0);
  let fy = ty_f - f32(ty0);

  let v00 = lut_lookup(ty0 * params.grid + tx0, bin);
  let v10 = lut_lookup(ty0 * params.grid + tx1, bin);
  let v01 = lut_lookup(ty1 * params.grid + tx0, bin);
  let v11 = lut_lookup(ty1 * params.grid + tx1, bin);

  let new_luma = v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy;
  let ratio = select(new_luma / max(luma, 1e-10), 1.0, luma < 1e-10);

  output[idx] = vec4<f32>(pixel.xyz * ratio, pixel.w);
}
"#;

/// Pyramid detail remap — per-level Laplacian coefficient remapping.
///
/// This shader handles a single level of the pyramid: subtracts upsampled
/// coarser level from current, remaps detail coefficients, adds back.
///
/// extra_buffers[0] = coarser level (upsampled to current dimensions)
pub const PYRAMID_REMAP_LEVEL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  sigma: f32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> fine: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> coarse_up: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let f = fine[idx];
  let c = coarse_up[idx];

  // Per-channel Laplacian coefficient remapping: d * sigma / (sigma + |d|)
  var result = vec4<f32>(0.0, 0.0, 0.0, f.w);
  let sigma = params.sigma;
  for (var ch = 0u; ch < 3u; ch++) {
    let fv = select(select(f.z, f.y, ch == 1u), f.x, ch == 0u);
    let cv = select(select(c.z, c.y, ch == 1u), c.x, ch == 0u);
    let detail = fv - cv;
    let remapped = select(detail * sigma / (sigma + abs(detail)), 0.0, abs(sigma) < 1e-10);
    let out_val = cv + remapped;
    if (ch == 0u) { result.x = out_val; }
    else if (ch == 1u) { result.y = out_val; }
    else { result.z = out_val; }
  }

  output[idx] = result;
}
"#;

/// Downsample 2x (for pyramid construction).
///
/// Writes to a reduction buffer at half resolution.
pub const DOWNSAMPLE_2X: &str = r#"
struct Params {
  in_width: u32,
  in_height: u32,
  out_width: u32,
  out_height: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.out_width * params.out_height) { return; }

  let ox = idx % params.out_width;
  let oy = idx / params.out_width;
  let sx = min(ox * 2u, params.in_width - 1u);
  let sy = min(oy * 2u, params.in_height - 1u);
  let sx1 = min(sx + 1u, params.in_width - 1u);
  let sy1 = min(sy + 1u, params.in_height - 1u);

  let p00 = input[sx + sy * params.in_width];
  let p10 = input[sx1 + sy * params.in_width];
  let p01 = input[sx + sy1 * params.in_width];
  let p11 = input[sx1 + sy1 * params.in_width];

  output[idx] = (p00 + p10 + p01 + p11) * 0.25;
}
"#;

/// Upsample 2x (for pyramid reconstruction).
pub const UPSAMPLE_2X: &str = r#"
struct Params {
  in_width: u32,
  in_height: u32,
  out_width: u32,
  out_height: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.out_width * params.out_height) { return; }

  let ox = idx % params.out_width;
  let oy = idx / params.out_width;
  let cx = min(ox / 2u, params.in_width - 1u);
  let cy = min(oy / 2u, params.in_height - 1u);

  output[idx] = input[cx + cy * params.in_width];
}
"#;

/// Retinex MSRCR color restoration + normalize.
///
/// extra_buffers[0] = original image
/// Reduction buffer (read) = MSR accumulator
pub const RETINEX_MSRCR_APPLY: &str = r#"
struct Params {
  width: u32,
  height: u32,
  num_scales: f32,
  alpha: f32,
  beta: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> original: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> accumulator: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let orig = original[idx];
  let msr = accumulator[idx].xyz / params.num_scales;

  // Color restoration
  let sum = orig.x + orig.y + orig.z;
  var result = vec3<f32>(0.0);
  for (var c = 0u; c < 3u; c++) {
    let channel = select(select(orig.z, orig.y, c == 1u), orig.x, c == 0u);
    let chromaticity = select(channel / sum, 1.0 / 3.0, sum < 1e-10);
    let color_gain = params.beta * max(log(params.alpha * chromaticity), -10.0);
    let msr_c = select(select(msr.z, msr.y, c == 1u), msr.x, c == 0u);
    if (c == 0u) { result.x = color_gain * msr_c; }
    else if (c == 1u) { result.y = color_gain * msr_c; }
    else { result.z = color_gain * msr_c; }
  }

  output[idx] = vec4<f32>(result, orig.w);
}
"#;
