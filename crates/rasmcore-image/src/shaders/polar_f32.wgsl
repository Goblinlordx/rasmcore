// f32 variant — auto-generated from U32Packed version
// Polar coordinate transform — Cartesian to polar mapping
// Matches IM DePolar convention: pixel-center coords, angle centered at w/2.
// Maps x -> angle, y -> radius in output polar image.

struct Params {
  width: u32,
  height: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
const PI: f32 = 3.14159265358979;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let wf = f32(params.width);
  let hf = f32(params.height);
  let cx = wf * 0.5;
  let cy = hf * 0.5;
  let max_radius = min(cx, cy);

  // Pixel-center convention
  let dx = f32(x) + 0.5;
  let dy = f32(y) + 0.5;

  // Invert depolar mapping: angle centered at w/2
  let angle = (dx - cx) / wf * 2.0 * PI;
  let radius = dy / hf * max_radius;

  // Source in Cartesian space (IM convention: sin for x, cos for y)
  let sx = cx + radius * sin(angle) - 0.5;
  let sy = cy + radius * cos(angle) - 0.5;

  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
