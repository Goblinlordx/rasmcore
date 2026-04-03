// 1D per-channel LUT shader — f32 with interpolation
// Applies a pre-fused per-channel lookup table (256 entries, stored as u32).
// Input: vec4<f32> in [0,1]. Scales to [0,255], does bilinear LUT lookup.

struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> lut: array<u32>;  // 256 u8 entries packed as u32

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];

  // Scale [0,1] to [0,255] and interpolate LUT
  let r_scaled = clamp(px.r * 255.0, 0.0, 255.0);
  let g_scaled = clamp(px.g * 255.0, 0.0, 255.0);
  let b_scaled = clamp(px.b * 255.0, 0.0, 255.0);

  let r_lo = u32(floor(r_scaled));
  let r_hi = min(r_lo + 1u, 255u);
  let r_frac = r_scaled - f32(r_lo);

  let g_lo = u32(floor(g_scaled));
  let g_hi = min(g_lo + 1u, 255u);
  let g_frac = g_scaled - f32(g_lo);

  let b_lo = u32(floor(b_scaled));
  let b_hi = min(b_lo + 1u, 255u);
  let b_frac = b_scaled - f32(b_lo);

  // Interpolated LUT lookup — smooth f32 output
  let nr = mix(f32(lut[r_lo]), f32(lut[r_hi]), r_frac) / 255.0;
  let ng = mix(f32(lut[g_lo]), f32(lut[g_hi]), g_frac) / 255.0;
  let nb = mix(f32(lut[b_lo]), f32(lut[b_hi]), b_frac) / 255.0;

  output[idx] = vec4<f32>(nr, ng, nb, px.a);
}
