// Charcoal sketch — Sobel edge detection + invert, f32 per-pixel
// Computes Sobel gradient magnitude on luminance, then inverts.
// Note: GPU version skips the optional pre/post blur for simplicity;
// the result is a clean edge-detected + inverted grayscale image
// written to all RGB channels.

struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn sample_luma(ix: i32, iy: i32) -> f32 {
  let sx = u32(clamp(ix, 0, i32(params.width) - 1));
  let sy = u32(clamp(iy, 0, i32(params.height) - 1));
  let px = input[sx + sy * params.width];
  // BT.601 luminance
  return 0.299 * px.x + 0.587 * px.y + 0.114 * px.z;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let ix = i32(x);
  let iy = i32(y);

  // Sample 3x3 neighborhood luminance
  let p00 = sample_luma(ix - 1, iy - 1);
  let p01 = sample_luma(ix, iy - 1);
  let p02 = sample_luma(ix + 1, iy - 1);
  let p10 = sample_luma(ix - 1, iy);
  let p12 = sample_luma(ix + 1, iy);
  let p20 = sample_luma(ix - 1, iy + 1);
  let p21 = sample_luma(ix, iy + 1);
  let p22 = sample_luma(ix + 1, iy + 1);

  // Sobel Gx = [-1,0,1; -2,0,2; -1,0,1]
  let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
  // Sobel Gy = [-1,-2,-1; 0,0,0; 1,2,1]
  let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

  // Gradient magnitude (L2 norm), scale to 0-1 range
  // Sobel max theoretical magnitude for 0-1 input is 4.0 (sqrt(16+16))
  let mag = clamp(sqrt(gx * gx + gy * gy), 0.0, 1.0);

  // Invert: charcoal = dark lines on white background
  let v = 1.0 - mag;

  let alpha = input[x + y * params.width].w;
  output[x + y * params.width] = vec4<f32>(v, v, v, alpha);
}
