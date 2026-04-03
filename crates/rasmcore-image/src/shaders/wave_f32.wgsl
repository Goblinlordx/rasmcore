// Wave — sinusoidal displacement, f32 variant

struct Params {
  width: u32,
  height: u32,
  amplitude: f32,
  wavelength: f32,
  vertical: f32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
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

  var sx: f32;
  var sy: f32;

  if (params.vertical > 0.5) {
    sx = f32(x) + params.amplitude * sin(2.0 * PI * f32(y) / params.wavelength);
    sy = f32(y);
  } else {
    sx = f32(x);
    sy = f32(y) + params.amplitude * sin(2.0 * PI * f32(x) / params.wavelength);
  }

  output[x + y * params.width] = sample_bilinear_f32(sx, sy);
}
