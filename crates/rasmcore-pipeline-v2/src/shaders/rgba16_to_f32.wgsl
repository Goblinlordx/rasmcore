// Convert packed u16 linear RGBA to f32 linear RGBA.
//
// Input: extra_buffers[0] as array<u32> — each u32 packs 2 u16 channels (little-endian).
//   Pixel N: packed_u16[N*2] = (R16 | G16<<16), packed_u16[N*2+1] = (B16 | A16<<16).
// Output: array<vec4<f32>> — f32 linear RGBA.
//
// Simple normalization: u16 / 65535.0. Already linear — no gamma correction.

struct Params {
  width: u32,
  height: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> packed_u16: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height;
  if (idx >= total) { return; }

  let rg = packed_u16[idx * 2u];
  let ba = packed_u16[idx * 2u + 1u];

  let r = f32(rg & 0xFFFFu) / 65535.0;
  let g = f32((rg >> 16u) & 0xFFFFu) / 65535.0;
  let b = f32(ba & 0xFFFFu) / 65535.0;
  let a = f32((ba >> 16u) & 0xFFFFu) / 65535.0;

  output[idx] = vec4<f32>(r, g, b, a);
}
