// Oil paint — neighborhood mode filter via intensity histogram, f32 per-pixel
// For each pixel, find the most frequent intensity bin in the neighborhood
// and output the average color of that bin.

struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

// Use 32 bins for GPU (full 256 would exceed register budget).
// This trades some precision for GPU feasibility but still produces
// a good oil-painting visual effect.
const NUM_BINS: u32 = 32u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let r = i32(params.radius);
  let w = params.width;
  let h = params.height;

  var count: array<u32, 32>;
  var sum_r: array<f32, 32>;
  var sum_g: array<f32, 32>;
  var sum_b: array<f32, 32>;

  // Zero the arrays
  for (var i = 0u; i < NUM_BINS; i++) {
    count[i] = 0u;
    sum_r[i] = 0.0;
    sum_g[i] = 0.0;
    sum_b[i] = 0.0;
  }

  let y0 = max(i32(y) - r, 0);
  let y1 = min(i32(y) + r + 1, i32(h));
  let x0 = max(i32(x) - r, 0);
  let x1 = min(i32(x) + r + 1, i32(w));

  for (var ny = y0; ny < y1; ny++) {
    for (var nx = x0; nx < x1; nx++) {
      let idx = u32(nx) + u32(ny) * w;
      let px = input[idx];
      // BT.601 luminance → bin index
      let luma = 0.299 * px.x + 0.587 * px.y + 0.114 * px.z;
      let bin = min(u32(luma * f32(NUM_BINS)), NUM_BINS - 1u);
      count[bin] += 1u;
      sum_r[bin] += px.x;
      sum_g[bin] += px.y;
      sum_b[bin] += px.z;
    }
  }

  // Find the bin with the highest count (mode)
  var max_bin = 0u;
  var max_count = 0u;
  for (var i = 0u; i < NUM_BINS; i++) {
    if (count[i] > max_count) {
      max_count = count[i];
      max_bin = i;
    }
  }

  let alpha = input[x + y * w].w;
  if (max_count > 0u) {
    let n = f32(max_count);
    output[x + y * w] = vec4<f32>(sum_r[max_bin] / n, sum_g[max_bin] / n, sum_b[max_bin] / n, alpha);
  } else {
    output[x + y * w] = input[x + y * w];
  }
}
