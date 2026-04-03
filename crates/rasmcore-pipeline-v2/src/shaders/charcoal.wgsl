// Charcoal — Sobel edge detection on luminance, inverted for sketch look.
// Note: blur pass is handled separately; this shader does edge detect + invert.

struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

const sobel_x = array<f32, 9>(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
const sobel_y = array<f32, 9>(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);

fn clamp_coord(v: i32, size: u32) -> u32 {
  return u32(clamp(v, 0, i32(size) - 1));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  var gx_val: f32 = 0.0;
  var gy_val: f32 = 0.0;
  for (var ky: i32 = 0; ky < 3; ky++) {
    for (var kx: i32 = 0; kx < 3; kx++) {
      let sx = clamp_coord(x + kx - 1, params.width);
      let sy = clamp_coord(y + ky - 1, params.height);
      let p = load_pixel(sy * params.width + sx);
      let luma = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
      let ki = ky * 3 + kx;
      gx_val += sobel_x[ki] * luma;
      gy_val += sobel_y[ki] * luma;
    }
  }
  let mag = min(sqrt(gx_val * gx_val + gy_val * gy_val), 1.0);
  // Invert: dark lines on white
  let v = 1.0 - mag;
  let idx = gid.y * params.width + gid.x;
  let orig = load_pixel(idx);
  store_pixel(idx, vec4<f32>(v, v, v, orig.w));
}
