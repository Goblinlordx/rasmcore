// Depolar — inverse polar coordinate transform (polar to Cartesian)
// Reverses the polar transform: maps (angle, radius) back to (x, y).

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

  // Source pixel (x, y) is in Cartesian space
  let dx = f32(x) - cx;
  let dy = f32(y) - cy;
  let radius = sqrt(dx * dx + dy * dy);
  var angle = atan2(dy, dx);
  if (angle < 0.0) {
    angle += 2.0 * PI;
  }

  // Map to polar image coordinates
  let sx = angle / (2.0 * PI) * f32(params.width);
  let sy = radius / max_radius * f32(params.height);

  output[x + y * params.width] = pack(sample_bilinear(sx, sy));
}
