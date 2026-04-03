// f32 variant — auto-generated from U32Packed version
// Spherize — per-pixel inverse coordinate transform with bilinear sampling
// Maps pixels through a spherical distortion centered on the image.

struct Params {
  width: u32,
  height: u32,
  amount: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let cx = f32(params.width) * 0.5;
  let cy = f32(params.height) * 0.5;
  let nx = (f32(x) - cx) / cx;
  let ny = (f32(y) - cy) / cy;
  let r = sqrt(nx * nx + ny * ny);

  var sx: f32;
  var sy: f32;
  if (r < 1.0 && r > 0.0) {
    let theta = asin(r) / r;
    let factor = mix(1.0, theta, params.amount);
    sx = nx * factor * cx + cx;
    sy = ny * factor * cy + cy;
  } else {
    sx = f32(x);
    sy = f32(y);
  }

  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
