// Guided filter — edge-preserving smoothing
// Simplified self-guided variant: guide = input image.
// Uses two-pass approach: pass 1 computes local means, pass 2 applies coefficients.
// For GPU simplicity, uses single-pass brute-force neighborhood (acceptable for small radii).

struct Params {
  width: u32,
  height: u32,
  radius: u32,
  epsilon: f32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let r = i32(params.radius);
  var mean_I = vec3<f32>(0.0);
  var mean_II = vec3<f32>(0.0);
  var count = 0.0;

  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let sx = clamp(i32(x) + dx, 0, i32(params.width) - 1);
      let sy = clamp(i32(y) + dy, 0, i32(params.height) - 1);
      let pixel = unpack(input[u32(sx) + u32(sy) * params.width]).xyz;
      mean_I += pixel;
      mean_II += pixel * pixel;
      count += 1.0;
    }
  }

  mean_I /= count;
  mean_II /= count;

  // Variance and coefficient
  let variance = mean_II - mean_I * mean_I;
  let a = variance / (variance + vec3<f32>(params.epsilon));
  let b = mean_I * (vec3<f32>(1.0) - a);

  let center = unpack(input[x + y * params.width]);
  let result = vec4<f32>(
    clamp(a.x * center.x + b.x, 0.0, 255.0),
    clamp(a.y * center.y + b.y, 0.0, 255.0),
    clamp(a.z * center.z + b.z, 0.0, 255.0),
    center.w,
  );

  output[x + y * params.width] = pack(result);
}
