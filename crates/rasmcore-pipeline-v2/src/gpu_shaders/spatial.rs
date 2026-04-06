//! GPU shaders for spatial (neighborhood) filters — f32 native.
//!
//! Separable blur shaders (GaussianBlur, BoxBlur) use two passes: H then V.
//! Kernel weights are passed via extra_buffers for variable-size kernels.

/// Gaussian blur — horizontal pass (separable).
///
/// Reads 1D kernel weights from extra_buffers[0] (f32 array, normalized).
/// `params.kernel_radius` determines the kernel half-width.
pub const GAUSSIAN_BLUR_H: &str = r#"
struct Params {
  width: u32,
  height: u32,
  kernel_radius: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = idx % params.width;
  let y = idx / params.width;
  let r = i32(params.kernel_radius);

  var sum = vec4<f32>(0.0);
  for (var dx = -r; dx <= r; dx++) {
    let sx = clamp(i32(x) + dx, 0, i32(params.width) - 1);
    let src_idx = u32(sx) + y * params.width;
    let ki = dx + r;
    sum += kernel[ki] * input[src_idx];
  }
  output[idx] = sum;
}
"#;

/// Gaussian blur — vertical pass (separable).
pub const GAUSSIAN_BLUR_V: &str = r#"
struct Params {
  width: u32,
  height: u32,
  kernel_radius: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = idx % params.width;
  let y = idx / params.width;
  let r = i32(params.kernel_radius);

  var sum = vec4<f32>(0.0);
  for (var dy = -r; dy <= r; dy++) {
    let sy = clamp(i32(y) + dy, 0, i32(params.height) - 1);
    let src_idx = x + u32(sy) * params.width;
    let ki = dy + r;
    sum += kernel[ki] * input[src_idx];
  }
  output[idx] = sum;
}
"#;

/// Box blur — horizontal pass.
pub const BOX_BLUR_H: &str = r#"
struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = idx % params.width;
  let y = idx / params.width;
  let r = i32(params.radius);
  let diameter = f32(2 * r + 1);

  var sum = vec4<f32>(0.0);
  for (var dx = -r; dx <= r; dx++) {
    let sx = clamp(i32(x) + dx, 0, i32(params.width) - 1);
    sum += input[u32(sx) + y * params.width];
  }
  output[idx] = sum / diameter;
}
"#;

/// Box blur — vertical pass.
pub const BOX_BLUR_V: &str = r#"
struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = idx % params.width;
  let y = idx / params.width;
  let r = i32(params.radius);
  let diameter = f32(2 * r + 1);

  var sum = vec4<f32>(0.0);
  for (var dy = -r; dy <= r; dy++) {
    let sy = clamp(i32(y) + dy, 0, i32(params.height) - 1);
    sum += input[x + u32(sy) * params.width];
  }
  output[idx] = sum / diameter;
}
"#;

/// Unsharp mask (sharpen) — applies after a blur pass.
///
/// Input is the blurred image (from previous pass); extra_buffers[0] has the original.
/// `output = original + amount * (original - blurred)`
pub const SHARPEN_APPLY: &str = r#"
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
  let detail = orig - blur;
  output[idx] = vec4<f32>(
    orig.x + params.amount * detail.x,
    orig.y + params.amount * detail.y,
    orig.z + params.amount * detail.z,
    orig.w,
  );
}
"#;

/// High-pass filter — subtracts blur from original, adding 0.5 offset.
///
/// Same pipeline as sharpen but `output = (original - blurred) + 0.5`
pub const HIGH_PASS_APPLY: &str = r#"
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

/// Median filter — 3x3/5x5/7x7 sorting network.
///
/// Uses a compile-time sorting network. For larger radii,
/// falls back to partial sort via register array.
pub const MEDIAN: &str = r#"
struct Params {
  width: u32,
  height: u32,
  radius: u32,
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

// Swap if a > b
fn sort2(a: ptr<function, f32>, b: ptr<function, f32>) {
  if (*a > *b) {
    let t = *a;
    *a = *b;
    *b = t;
  }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let r = i32(params.radius);
  let idx = gid.x + gid.y * params.width;

  // Per-channel median via insertion into a register array
  // Max neighborhood: (2*radius+1)^2. Limit radius to 3 for register pressure.
  var result = vec4<f32>(0.0, 0.0, 0.0, get_pixel(x, y).w);

  for (var c = 0u; c < 3u; c++) {
    // Collect neighborhood values
    var vals: array<f32, 49>; // max 7x7
    var count = 0u;
    for (var dy = -r; dy <= r; dy++) {
      for (var dx = -r; dx <= r; dx++) {
        let p = get_pixel(x + dx, y + dy);
        let v = select(select(p.z, p.y, c == 1u), p.x, c == 0u);
        vals[count] = v;
        count++;
        if (count >= 49u) { break; }
      }
      if (count >= 49u) { break; }
    }

    // Partial selection sort to find median
    let mid = count / 2u;
    for (var i = 0u; i <= mid; i++) {
      var min_idx = i;
      for (var j = i + 1u; j < count; j++) {
        if (vals[j] < vals[min_idx]) { min_idx = j; }
      }
      if (min_idx != i) {
        let t = vals[i];
        vals[i] = vals[min_idx];
        vals[min_idx] = t;
      }
    }

    if (c == 0u) { result.x = vals[mid]; }
    else if (c == 1u) { result.y = vals[mid]; }
    else { result.z = vals[mid]; }
  }

  output[idx] = result;
}
"#;

/// General NxN convolution — kernel passed via extra_buffers.
///
/// extra_buffers[0]: f32 kernel weights (kernel_w × kernel_h)
pub const CONVOLVE: &str = r#"
struct Params {
  width: u32,
  height: u32,
  kernel_w: u32,
  kernel_h: u32,
  inv_divisor: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let rw = i32(params.kernel_w / 2u);
  let rh = i32(params.kernel_h / 2u);
  let idx = gid.x + gid.y * params.width;

  var sum = vec4<f32>(0.0);
  for (var ky = 0; ky < i32(params.kernel_h); ky++) {
    for (var kx = 0; kx < i32(params.kernel_w); kx++) {
      let sx = clamp(x + kx - rw, 0, i32(params.width) - 1);
      let sy = clamp(y + ky - rh, 0, i32(params.height) - 1);
      let k = kernel[ky * i32(params.kernel_w) + kx];
      sum += k * input[u32(sx) + u32(sy) * params.width];
    }
  }
  output[idx] = sum * params.inv_divisor;
}
"#;

/// Bilateral filter — edge-preserving smoothing.
pub const BILATERAL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
  inv_sigma_color_sq: f32, // -0.5 / (sigma_color^2)
  inv_sigma_space_sq: f32, // -0.5 / (sigma_space^2)
  _pad2: u32,
  _pad3: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let r = i32(params.radius);
  let idx = gid.x + gid.y * params.width;
  let center = input[idx];

  var sum = vec3<f32>(0.0);
  var weight_sum = 0.0;

  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let sx = clamp(x + dx, 0, i32(params.width) - 1);
      let sy = clamp(y + dy, 0, i32(params.height) - 1);
      let neighbor = input[u32(sx) + u32(sy) * params.width];

      let dist2 = f32(dx * dx + dy * dy);
      let ws = exp(dist2 * params.inv_sigma_space_sq);

      let d = neighbor.xyz - center.xyz;
      let cd2 = dot(d, d);
      let wc = exp(cd2 * params.inv_sigma_color_sq);

      let w = ws * wc;
      sum += w * neighbor.xyz;
      weight_sum += w;
    }
  }

  let inv_w = select(1.0 / weight_sum, 0.0, weight_sum < 1e-10);
  output[idx] = vec4<f32>(sum * inv_w, center.w);
}
"#;

/// Motion blur — directional line sampling.
pub const MOTION_BLUR: &str = r#"
struct Params {
  width: u32,
  height: u32,
  steps: u32,
  _pad: u32,
  dx: f32, // cos(angle)
  dy: f32, // sin(angle)
  inv_steps: f32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = f32(idx % params.width);
  let y = f32(idx / params.width);
  let half = f32(params.steps) * 0.5;

  var sum = vec4<f32>(0.0);
  for (var s = 0u; s <= params.steps; s++) {
    let t = f32(s) - half;
    let sx = clamp(i32(round(x + t * params.dx)), 0, i32(params.width) - 1);
    let sy = clamp(i32(round(y + t * params.dy)), 0, i32(params.height) - 1);
    sum += input[u32(sx) + u32(sy) * params.width];
  }
  output[idx] = sum * params.inv_steps;
}
"#;

/// Displacement map — per-pixel warp with bilinear interpolation.
///
/// extra_buffers[0]: f32 pairs [map_x, map_y] for each pixel (2 × w × h floats).
pub const DISPLACEMENT_MAP: &str = r#"
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> disp_map: array<f32>;

fn sample_bilinear(sx: f32, sy: f32) -> vec4<f32> {
  let x0 = i32(floor(sx));
  let y0 = i32(floor(sy));
  let fx = sx - f32(x0);
  let fy = sy - f32(y0);

  let w = i32(params.width);
  let h = i32(params.height);

  let cx0 = clamp(x0, 0, w - 1);
  let cy0 = clamp(y0, 0, h - 1);
  let cx1 = clamp(x0 + 1, 0, w - 1);
  let cy1 = clamp(y0 + 1, 0, h - 1);

  let p00 = input[u32(cx0) + u32(cy0) * params.width];
  let p10 = input[u32(cx1) + u32(cy0) * params.width];
  let p01 = input[u32(cx0) + u32(cy1) * params.width];
  let p11 = input[u32(cx1) + u32(cy1) * params.width];

  return p00 * (1.0 - fx) * (1.0 - fy)
       + p10 * fx * (1.0 - fy)
       + p01 * (1.0 - fx) * fy
       + p11 * fx * fy;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let sx = disp_map[idx * 2u];
  let sy = disp_map[idx * 2u + 1u];
  output[idx] = sample_bilinear(sx, sy);
}
"#;

// ─── Advanced Spatial Shaders ────────────────────────────────────────────────

/// Zoom blur — radial blur from center.
///
/// Samples along radial lines from each pixel toward center_x/center_y.
pub const ZOOM_BLUR: &str = r#"
struct Params {
  width: u32,
  height: u32,
  samples: u32,
  _pad: u32,
  center_x: f32,
  center_y: f32,
  factor: f32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = f32(idx % params.width);
  let y = f32(idx / params.width);
  let dx = params.center_x - x;
  let dy = params.center_y - y;
  let n = params.samples;
  let inv_n = 1.0 / f32(n);

  var sum = vec4<f32>(0.0);
  for (var s = 0u; s < n; s++) {
    let t = f32(s) * params.factor / f32(n);
    let sx = clamp(i32(round(x + dx * t)), 0, i32(params.width) - 1);
    let sy = clamp(i32(round(y + dy * t)), 0, i32(params.height) - 1);
    sum += input[u32(sx) + u32(sy) * params.width];
  }
  output[idx] = sum * inv_n;
}
"#;

/// Spin blur — rotational blur around center.
///
/// Samples along circular arcs around center_x/center_y.
pub const SPIN_BLUR: &str = r#"
struct Params {
  width: u32,
  height: u32,
  samples: u32,
  _pad: u32,
  center_x: f32,
  center_y: f32,
  angle_rad: f32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = f32(idx % params.width);
  let y = f32(idx / params.width);
  let dx = x - params.center_x;
  let dy = y - params.center_y;
  let n = params.samples;
  let inv_n = 1.0 / f32(n);
  let half = f32(n) * 0.5;

  var sum = vec4<f32>(0.0);
  for (var s = 0u; s < n; s++) {
    let offset = params.angle_rad * (f32(s) - half) / f32(n);
    let cos_t = cos(offset);
    let sin_t = sin(offset);
    let sx = clamp(i32(round(params.center_x + dx * cos_t - dy * sin_t)), 0, i32(params.width) - 1);
    let sy = clamp(i32(round(params.center_y + dx * sin_t + dy * cos_t)), 0, i32(params.height) - 1);
    sum += input[u32(sx) + u32(sy) * params.width];
  }
  output[idx] = sum * inv_n;
}
"#;

/// Tilt shift — blend original with blurred based on distance from focus band.
///
/// Input is the blurred image; extra_buffers[0] has the original.
/// Computes per-pixel smoothstep mask from distance to focus band.
pub const TILT_SHIFT_BLEND: &str = r#"
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
  focus_position: f32,
  half_band: f32,
  transition: f32,
  cos_angle: f32,
  sin_angle: f32,
  _pad2: u32,
  _pad3: u32,
  _pad4: u32,
}
@group(0) @binding(0) var<storage, read> blurred: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> original: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let x = f32(idx % params.width) / f32(params.width) - 0.5;
  let y = f32(idx / params.width) / f32(params.height) - params.focus_position;
  let dist = abs(x * params.sin_angle + y * params.cos_angle);

  var t: f32;
  if (dist < params.half_band) {
    t = 0.0;
  } else {
    let d = min((dist - params.half_band) / params.transition, 1.0);
    t = d * d * (3.0 - 2.0 * d); // smoothstep
  }

  let orig = original[idx];
  let blur = blurred[idx];
  output[idx] = mix(orig, blur, vec4<f32>(t));
}
"#;

/// Convolution with kernel from extra_buffer — used by lens_blur and bokeh_blur.
///
/// Same as CONVOLVE but separate constant for clarity.
/// Kernel is in extra_buffers[0] as f32 array.
pub const LENS_BLUR: &str = CONVOLVE;
