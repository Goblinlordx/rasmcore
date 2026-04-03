// Levels — remap input range with gamma, f32 per-pixel
// black/white are normalized [0,1], inv_gamma = 1/gamma

struct Params {
  width: u32,
  height: u32,
  black: f32,
  range_inv: f32,
  inv_gamma: f32,
  _pad0: u32,
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
  let idx = x + y * params.width;
  let px = input[idx];
  let normalized = clamp((px.rgb - params.black) * params.range_inv, vec3<f32>(0.0), vec3<f32>(1.0));
  let v = pow(normalized, vec3<f32>(params.inv_gamma));
  output[idx] = vec4<f32>(v, px.a);
}
