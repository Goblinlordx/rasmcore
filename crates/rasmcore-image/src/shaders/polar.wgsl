// Polar coordinate transform — Cartesian to polar mapping
// Maps x -> angle (0..2PI), y -> radius (0..max_radius)

struct Params {
  width: u32,
  height: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
const PI: f32 = 3.14159265358979;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let cx = f32(params.width) * 0.5;
  let cy = f32(params.height) * 0.5;
  let max_radius = sqrt(cx * cx + cy * cy);

  // Output pixel (x, y) maps to polar coords:
  // angle = x / width * 2PI
  // radius = y / height * max_radius
  let angle = f32(x) / f32(params.width) * 2.0 * PI;
  let radius = f32(y) / f32(params.height) * max_radius;

  // Convert back to source Cartesian coords
  let sx = radius * cos(angle) + cx;
  let sy = radius * sin(angle) + cy;

  output[x + y * params.width] = pack(sample_bilinear(sx, sy));
}
