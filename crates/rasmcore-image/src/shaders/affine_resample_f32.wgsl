// Fused affine transform + bilinear resample — f32 variant
// Uses array<vec4<f32>> storage buffers for full-precision data.
// Applies a 3x3 inverse matrix to each output pixel to find the source coordinate,
// then bilinear-samples from the input. One dispatch replaces separate
// resize + rotate + flip + crop passes.

struct Params {
  src_width: u32,
  src_height: u32,
  dst_width: u32,
  dst_height: u32,
  // Inverse affine matrix (row-major, 2x3):
  //   sx = inv_a * x + inv_b * y + inv_tx
  //   sy = inv_c * x + inv_d * y + inv_ty
  inv_a: f32,
  inv_b: f32,
  inv_tx: f32,
  inv_c: f32,
  inv_d: f32,
  inv_ty: f32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
fn sample_bilinear_src_f32(fx: f32, fy: f32) -> vec4<f32> {
  // Out-of-bounds: transparent black
  if (fx < -0.5 || fx >= f32(params.src_width) - 0.5 ||
      fy < -0.5 || fy >= f32(params.src_height) - 0.5) {
    return vec4<f32>(0.0);
  }

  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let dx = fx - f32(ix);
  let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.src_width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.src_width) - 1);
  let y0 = clamp(iy, 0, i32(params.src_height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.src_height) - 1);
  let p00 = input[u32(x0) + u32(y0) * params.src_width];
  let p10 = input[u32(x1) + u32(y0) * params.src_width];
  let p01 = input[u32(x0) + u32(y1) * params.src_width];
  let p11 = input[u32(x1) + u32(y1) * params.src_width];
  return mix(mix(p00, p10, vec4<f32>(dx)), mix(p01, p11, vec4<f32>(dx)), vec4<f32>(dy));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.dst_width || y >= params.dst_height) { return; }

  // Apply inverse matrix to find source coordinate
  let fx = f32(x);
  let fy = f32(y);
  let sx = params.inv_a * fx + params.inv_b * fy + params.inv_tx;
  let sy = params.inv_c * fx + params.inv_d * fy + params.inv_ty;

  output[x + y * params.dst_width] = sample_bilinear_src_f32(sx, sy);
}
