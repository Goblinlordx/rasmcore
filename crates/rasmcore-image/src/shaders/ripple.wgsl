// Ripple — concentric wave distortion from a center point

struct Params {
  width: u32,
  height: u32,
  amplitude: f32,
  wavelength: f32,
  center_x: f32,
  center_y: f32,
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

  let cx = params.center_x * f32(params.width);
  let cy = params.center_y * f32(params.height);
  let dx = f32(x) - cx;
  let dy = f32(y) - cy;
  let dist = sqrt(dx * dx + dy * dy);

  // Displacement along radial direction
  let displacement = params.amplitude * sin(2.0 * PI * dist / params.wavelength);
  var sx: f32;
  var sy: f32;
  if (dist > 0.0) {
    sx = f32(x) + displacement * dx / dist;
    sy = f32(y) + displacement * dy / dist;
  } else {
    sx = f32(x);
    sy = f32(y);
  }

  output[x + y * params.width] = pack(sample_bilinear(sx, sy));
}
