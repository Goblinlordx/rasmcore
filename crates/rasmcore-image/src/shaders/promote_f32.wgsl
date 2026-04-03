// Promote u8 (packed u32 RGBA8) to f32 (vec4<f32>) on GPU.
// Reduces CPU→GPU transfer by 4x: upload 4 bytes/pixel instead of 16.
//
// Input: extra_buffers[0] = array<u32> packed RGBA8 source pixels
// Output: array<vec4<f32>> promoted f32 pixels (binding 1)
// Binding 0 (input) is unused — the source data is in the extra buffer.

@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> source_u8: array<u32>;

struct Params {
  width: u32,
  height: u32,
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let packed = source_u8[idx];
  let r = f32((packed >> 0u) & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  let a = f32((packed >> 24u) & 0xFFu) / 255.0;
  output[idx] = vec4<f32>(r, g, b, a);
}
