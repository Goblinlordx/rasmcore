// Pixelate — block-grid mosaic. Each pixel reads from its block center.
// Block averaging is approximated by sampling the center pixel of each block.

struct Params {
  width: u32,
  height: u32,
  block_size: u32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let bs = params.block_size;
  // Find center of this pixel's block
  let block_x = (gid.x / bs) * bs + bs / 2u;
  let block_y = (gid.y / bs) * bs + bs / 2u;
  let cx = min(block_x, params.width - 1u);
  let cy = min(block_y, params.height - 1u);
  let center = load_pixel(cy * params.width + cx);
  let idx = gid.y * params.width + gid.x;
  let orig = load_pixel(idx);
  // Preserve alpha from original pixel
  store_pixel(idx, vec4<f32>(center.x, center.y, center.z, orig.w));
}
