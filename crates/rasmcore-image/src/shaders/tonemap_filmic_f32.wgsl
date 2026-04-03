// Filmic (ACES/Narkowicz) tone mapping — f32 per-pixel
// out = (x*(a*x+b)) / (x*(c*x+d)+e) per channel, alpha preserved

struct Params {
  width: u32,
  height: u32,
  a: f32,
  b: f32,
  c: f32,
  d: f32,
  e: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn filmic(v: vec3<f32>) -> vec3<f32> {
  let a = params.a;
  let b = params.b;
  let c = params.c;
  let d = params.d;
  let e = params.e;
  return (v * (a * v + vec3<f32>(b))) / (v * (c * v + vec3<f32>(d)) + vec3<f32>(e));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = unpack_f32(input[idx]);
  let mapped = filmic(px.rgb);
  output[idx] = pack_f32(vec4<f32>(mapped, px.a));
}
