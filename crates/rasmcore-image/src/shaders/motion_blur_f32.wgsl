// f32 variant
// Motion blur — directional linear blur at given angle
// Samples N taps along a line at the specified angle.

struct Params {
  width: u32,
  height: u32,
  length: u32,
  angle: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let dx = cos(params.angle);
  let dy = sin(params.angle);
  let half_len = f32(params.length);
  let samples = 2u * params.length + 1u;
  let inv_samples = 1.0 / f32(samples);

  var sum = vec4<f32>(0.0);
  let len = i32(params.length);
  for (var i = -len; i <= len; i++) {
    let fx = f32(x) + f32(i) * dx;
    let fy = f32(y) + f32(i) * dy;
    sum += sample_bilinear_f32(fx, fy);
  }

  output[x + y * params.width] = sum * inv_samples;
}
