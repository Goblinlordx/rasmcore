// Emboss — 3D relief via fixed 3x3 directional kernel + midgray offset.

struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

const kernel = array<f32, 9>(-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0);

fn clamp_coord(v: i32, size: u32) -> u32 {
  return u32(clamp(v, 0, i32(size) - 1));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  var sum = vec3<f32>(0.0, 0.0, 0.0);
  for (var ky: i32 = 0; ky < 3; ky++) {
    for (var kx: i32 = 0; kx < 3; kx++) {
      let sx = clamp_coord(x + kx - 1, params.width);
      let sy = clamp_coord(y + ky - 1, params.height);
      let p = load_pixel(sy * params.width + sx);
      let k = kernel[ky * 3 + kx];
      sum += vec3<f32>(p.x * k, p.y * k, p.z * k);
    }
  }
  let idx = gid.y * params.width + gid.x;
  let orig = load_pixel(idx);
  store_pixel(idx, vec4<f32>(sum.x + 0.5, sum.y + 0.5, sum.z + 0.5, orig.w));
}
