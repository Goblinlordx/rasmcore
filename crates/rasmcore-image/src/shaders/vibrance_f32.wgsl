// Vibrance — f32 per-pixel
// Selectively boosts low-saturation pixels, leaves high-saturation alone
// amount = -1..1

struct Params {
  width: u32,
  height: u32,
  amount: f32,
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
  let mx = max(max(px.r, px.g), px.b);
  let mn = min(min(px.r, px.g), px.b);
  let sat = mx - mn;
  let amt = params.amount * (1.0 - sat) * (1.0 - sat);
  let luma = 0.2126 * px.r + 0.7152 * px.g + 0.0722 * px.b;
  let v = clamp(mix(vec3<f32>(luma), px.rgb, 1.0 + amt), vec3<f32>(0.0), vec3<f32>(1.0));
  output[idx] = vec4<f32>(v, px.a);
}
