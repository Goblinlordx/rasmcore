// Convert packed u8 sRGB RGBA to f32 linear RGBA.
//
// Input: extra_buffers[0] as array<u32> — each u32 packs 4 u8 channels (R|G|B|A).
// Output: array<vec4<f32>> — f32 linear RGBA.
//
// Applies IEC 61966-2-1 sRGB EOTF (inverse gamma) to RGB channels.
// Alpha is normalized without gamma correction.

struct Params {
  width: u32,
  height: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> packed_u8: array<u32>;

fn srgb_to_linear(v: f32) -> f32 {
  if (v <= 0.04045) {
    return v / 12.92;
  }
  return pow((v + 0.055) / 1.055, 2.4);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height;
  if (idx >= total) { return; }

  let packed = packed_u8[idx];
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  let a = f32((packed >> 24u) & 0xFFu) / 255.0;

  output[idx] = vec4<f32>(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a);
}
