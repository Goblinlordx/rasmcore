// Channel mixer — 3x3 RGB matrix multiply, f32 per-pixel

struct Params {
  width: u32,
  height: u32,
  rr: f32,
  rg: f32,
  rb: f32,
  gr: f32,
  gg: f32,
  gb: f32,
  br: f32,
  bg: f32,
  bb: f32,
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
  let r = clamp(px.r * params.rr + px.g * params.rg + px.b * params.rb, 0.0, 1.0);
  let g = clamp(px.r * params.gr + px.g * params.gg + px.b * params.gb, 0.0, 1.0);
  let b = clamp(px.r * params.br + px.g * params.bg + px.b * params.bb, 0.0, 1.0);
  output[idx] = vec4<f32>(r, g, b, px.a);
}
