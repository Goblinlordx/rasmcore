// Oil paint — neighborhood mode filter. For each pixel, find the most
// frequent intensity bin in the neighborhood and output that bin's average color.

struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

const BINS: u32 = 32u; // Reduced from 256 for GPU register pressure

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let r = i32(params.radius);

  // Per-bin counters and color accumulators (in registers)
  var count: array<u32, 32>;
  var sum_r: array<f32, 32>;
  var sum_g: array<f32, 32>;
  var sum_b: array<f32, 32>;
  for (var i: u32 = 0u; i < BINS; i++) {
    count[i] = 0u;
    sum_r[i] = 0.0;
    sum_g[i] = 0.0;
    sum_b[i] = 0.0;
  }

  let y0 = max(y - r, 0);
  let y1 = min(y + r + 1, i32(params.height));
  let x0 = max(x - r, 0);
  let x1 = min(x + r + 1, i32(params.width));

  for (var ny = y0; ny < y1; ny++) {
    for (var nx = x0; nx < x1; nx++) {
      let p = load_pixel(u32(ny) * params.width + u32(nx));
      let luma = clamp(0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z, 0.0, 1.0);
      let bin = min(u32(luma * f32(BINS - 1u) + 0.5), BINS - 1u);
      count[bin] += 1u;
      sum_r[bin] += p.x;
      sum_g[bin] += p.y;
      sum_b[bin] += p.z;
    }
  }

  // Find mode bin
  var max_count: u32 = 0u;
  var max_bin: u32 = 0u;
  for (var i: u32 = 0u; i < BINS; i++) {
    if (count[i] > max_count) {
      max_count = count[i];
      max_bin = i;
    }
  }

  let cnt = f32(max(max_count, 1u));
  let idx = gid.y * params.width + gid.x;
  let orig = load_pixel(idx);
  store_pixel(idx, vec4<f32>(sum_r[max_bin] / cnt, sum_g[max_bin] / cnt, sum_b[max_bin] / cnt, orig.w));
}
