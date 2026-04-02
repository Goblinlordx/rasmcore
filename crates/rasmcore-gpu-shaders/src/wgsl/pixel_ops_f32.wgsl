// ── Shared pixel identity ops for f32 vec4 buffers ───────────────────────────
// When using array<vec4<f32>> storage buffers, pack/unpack are identity:
// the data is already in vec4<f32> format, no conversion needed.
// These functions exist so shaders can use the same call pattern regardless
// of buffer format (u32-packed vs f32-vec4).

fn unpack_f32(pixel: vec4<f32>) -> vec4<f32> {
  return pixel;
}

fn pack_f32(color: vec4<f32>) -> vec4<f32> {
  return color;
}
