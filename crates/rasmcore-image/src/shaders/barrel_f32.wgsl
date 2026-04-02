// Barrel/pincushion distortion — f32 vec4 variant
// k1 > 0: barrel distortion, k1 < 0: pincushion distortion

struct Params {
  width: u32,
  height: u32,
  k1: f32,
  k2: f32,
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
  let r2 = nx * nx + ny * ny;
  let r4 = r2 * r2;

  let distortion = 1.0 + params.k1 * r2 + params.k2 * r4;
  let sx = nx * distortion * cx + cx;
  let sy = ny * distortion * cy + cy;

  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
