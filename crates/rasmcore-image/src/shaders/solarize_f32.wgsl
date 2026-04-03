// Solarize — invert pixels above threshold
// threshold is normalized [0,1]

struct Params {
  width: u32,
  height: u32,
  threshold: f32,
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
  let t = params.threshold;
  let r = select(px.r, 1.0 - px.r, px.r > t);
  let g = select(px.g, 1.0 - px.g, px.g > t);
  let b = select(px.b, 1.0 - px.b, px.b > t);
  output[idx] = vec4<f32>(r, g, b, px.a);
}
