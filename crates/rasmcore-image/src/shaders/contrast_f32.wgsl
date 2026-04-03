// Contrast adjustment — f32 per-pixel
// factor = contrast multiplier around midpoint 0.5

struct Params {
  width: u32,
  height: u32,
  factor: f32,
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
  let idx = x + y * params.width;
  let px = input[idx];
  let v = clamp((px.rgb - 0.5) * params.factor + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
  output[idx] = vec4<f32>(v, px.a);
}
