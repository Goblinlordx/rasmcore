// f32 variant — auto-generated from U32Packed version
// Depolar — inverse polar coordinate transform (polar to Cartesian)
// Matches IM Polar convention: pixel-center coords, atan2(dx, dy) for angle.
// Maps Cartesian output -> polar source coordinates.

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
  let dx = f32(x) + 0.5 - cx;
  let dy = f32(y) + 0.5 - cy;
  let radius = sqrt(dx * dx + dy * dy);

  // IM convention: atan2(horizontal, vertical) — matches CPU atan2(ii, jj)
  var angle = atan2(dx, dy);

  // Normalize angle to [-0.5, 0.5) turns and map to x-coordinate
  var xx = angle / (2.0 * PI);
  xx = xx - round(xx);
  let sx = xx * wf + cx - 0.5;
  let sy = radius * (hf / max_radius) - 0.5;

  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
