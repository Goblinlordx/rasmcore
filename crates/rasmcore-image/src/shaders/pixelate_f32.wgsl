// Pixelate — block-grid mosaic averaging, f32 per-pixel
// Tiles a fixed grid from (0,0), averages each cell, fills with mean color.

struct Params {
  width: u32,
  height: u32,
  block_size: u32,
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

  let bs = params.block_size;

  // Compute the top-left corner of this pixel's block
  let bx = (x / bs) * bs;
  let by = (y / bs) * bs;

  // Compute block extents (truncated at image bounds)
  let bw = min(bs, params.width - bx);
  let bh = min(bs, params.height - by);
  let count = f32(bw * bh);

  // Accumulate block average
  var sum = vec4<f32>(0.0);
  for (var dy = 0u; dy < bh; dy++) {
    for (var dx = 0u; dx < bw; dx++) {
      let idx = (bx + dx) + (by + dy) * params.width;
      sum += input[idx];
    }
  }
  let avg = sum / count;

  // Preserve alpha from the original pixel
  let alpha = input[x + y * params.width].w;
  output[x + y * params.width] = vec4<f32>(avg.xyz, alpha);
}
