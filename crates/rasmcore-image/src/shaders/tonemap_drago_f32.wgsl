// Drago logarithmic tone mapping — f32 per-pixel
// out = ln(1 + x) / ln(1 + l_max) with bias, alpha preserved

struct Params {
  width: u32,
  height: u32,
  l_max: f32,
  bias: f32,
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
  let px = unpack_f32(input[idx]);

  let denom = log(1.0 + params.l_max);
  let bias = params.bias;
  // Drago biased log mapping: log(1 + x * bias) / log(1 + l_max)
  let mapped = vec3<f32>(
    log(1.0 + px.r * bias) / denom,
    log(1.0 + px.g * bias) / denom,
    log(1.0 + px.b * bias) / denom,
  );

  output[idx] = pack_f32(vec4<f32>(mapped, px.a));
}
