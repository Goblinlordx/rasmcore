// 1D per-channel LUT shader — single lookup per channel
// Applies a pre-fused per-channel lookup table (256 entries).

struct Params {
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> lut: array<u32>;  // 256 u8->u8 entries packed as u32

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.width + gid.x;
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  let pixel = input[idx];
  let r = pixel & 0xFFu;
  let g = (pixel >> 8u) & 0xFFu;
  let b = (pixel >> 16u) & 0xFFu;
  let a = pixel >> 24u;

  // Per-channel LUT lookup (same LUT for all channels — fused point ops)
  let nr = lut[r];
  let ng = lut[g];
  let nb = lut[b];

  output[idx] = nr | (ng << 8u) | (nb << 16u) | (a << 24u);
}
