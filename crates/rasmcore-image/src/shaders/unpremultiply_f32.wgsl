// Unpremultiply alpha — f32 per-pixel
// if a > 0: out.rgb = in.rgb / in.a; out.a = in.a, else out = vec4(0)

struct Params {
  width: u32,
  height: u32,
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
  if (px.a > 0.0) {
    output[idx] = vec4<f32>(px.rgb / px.a, px.a);
  } else {
    output[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
}
