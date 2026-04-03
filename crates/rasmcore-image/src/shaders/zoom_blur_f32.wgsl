// f32 variant
// Zoom blur — radial streak from a center point
// Matches GIMP/GEGL zoom motion blur: pixels further from center get longer streaks.

struct Params {
  width: u32,
  height: u32,
  center_x: f32,
  center_y: f32,
  factor: f32,
  samples: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let cx = params.center_x * f32(params.width);
  let cy = params.center_y * f32(params.height);

  // Direction from center to pixel
  let dx = f32(x) - cx;
  let dy = f32(y) - cy;

  var sum = vec4<f32>(0.0);
  let n = f32(params.samples);

  for (var i = 0u; i < params.samples; i++) {
    let t = (f32(i) / n - 0.5) * params.factor;
    let fx = f32(x) + dx * t;
    let fy = f32(y) + dy * t;
    sum += sample_bilinear_f32(fx, fy);
  }

  output[x + y * params.width] = sum / n;
}
