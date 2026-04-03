// Sigmoidal contrast — f32 per-pixel
// S-curve: sig(x) = 1 / (1 + exp(-strength * (x - midpoint)))
// sharpen: output = (sig(x) - sig(0)) / (sig(1) - sig(0))
// soften: inverse of the above
// strength = contrast amount, midpoint = center point (0-1)
// is_sharpen = 1.0 for sharpen, 0.0 for soften

struct Params {
  width: u32,
  height: u32,
  strength: f32,
  midpoint: f32,
  sig_0: f32,
  sig_range_inv: f32,
  sig_range: f32,
  is_sharpen: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn apply_sharpen(x: f32) -> f32 {
  let sig_x = 1.0 / (1.0 + exp(-params.strength * (x - params.midpoint)));
  return clamp((sig_x - params.sig_0) * params.sig_range_inv, 0.0, 1.0);
}

fn apply_soften(x: f32) -> f32 {
  let y_scaled = x * params.sig_range + params.sig_0;
  let y_clamped = clamp(y_scaled, 1e-7, 1.0 - 1e-7);
  return clamp(params.midpoint - log((1.0 - y_clamped) / y_clamped) / params.strength, 0.0, 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];
  var v: vec3<f32>;
  if (params.is_sharpen > 0.5) {
    v = vec3<f32>(apply_sharpen(px.r), apply_sharpen(px.g), apply_sharpen(px.b));
  } else {
    v = vec3<f32>(apply_soften(px.r), apply_soften(px.g), apply_soften(px.b));
  }
  output[idx] = vec4<f32>(v, px.a);
}
