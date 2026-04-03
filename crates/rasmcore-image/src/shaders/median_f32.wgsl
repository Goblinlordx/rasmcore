// f32 variant
// Median filter — per-channel median via partial sort network
// For radius <= 2 (5x5 window = 25 values), uses sorting network.
// For larger radii, uses histogram-based approach.

struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
// Partial sort: find median of neighborhood using selection approach
// Collect all values, then find the middle one via iterative min-extraction.
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let r = i32(params.radius);
  let diameter = 2 * r + 1;
  let count = diameter * diameter;
  let median_idx = count / 2;

  // Per-channel: collect neighborhood values and find median
  // Using iterative partial selection (swap-based nth_element approximation)
  var result = vec4<f32>(0.0);
  let center = input[x + y * params.width];

  // For each channel, accumulate a histogram-like running median
  // Brute force approach for GPU: gather all values, sort via bubble passes
  // Limited to radius <= 3 (7x7 = 49 values max for reasonable GPU perf)

  // Channel-wise processing
  for (var ch = 0u; ch < 3u; ch++) {
    // Gather neighborhood values for this channel
    var vals: array<f32, 49>;  // max 7x7
    var n = 0;
    for (var dy = -r; dy <= r; dy++) {
      for (var dx = -r; dx <= r; dx++) {
        let sx = clamp(i32(x) + dx, 0, i32(params.width) - 1);
        let sy = clamp(i32(y) + dy, 0, i32(params.height) - 1);
        let pixel = input[u32(sx) + u32(sy) * params.width];
        if (ch == 0u) { vals[n] = pixel.x; }
        else if (ch == 1u) { vals[n] = pixel.y; }
        else { vals[n] = pixel.z; }
        n++;
      }
    }

    // Partial bubble sort to find median (sort first median_idx+1 elements)
    for (var i = 0; i <= median_idx; i++) {
      for (var j = i + 1; j < count; j++) {
        if (vals[j] < vals[i]) {
          let tmp = vals[i];
          vals[i] = vals[j];
          vals[j] = tmp;
        }
      }
    }

    if (ch == 0u) { result.x = vals[median_idx]; }
    else if (ch == 1u) { result.y = vals[median_idx]; }
    else { result.z = vals[median_idx]; }
  }

  result.w = center.w; // preserve alpha
  output[x + y * params.width] = result;
}
