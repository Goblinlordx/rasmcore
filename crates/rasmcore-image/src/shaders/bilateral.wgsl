// Bilateral filter — spatial + range weighted neighborhood
// O(radius^2) per pixel with per-sample spatial and color distance weighting.

struct Params {
  width: u32,
  height: u32,
  radius: u32,
  sigma_spatial: f32,
  sigma_range: f32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let center = unpack(input[x + y * params.width]);
  var sum = vec4<f32>(0.0);
  var weight_sum = 0.0;
  let r = i32(params.radius);
  let inv_2ss = -0.5 / (params.sigma_spatial * params.sigma_spatial);
  let inv_2sr = -0.5 / (params.sigma_range * params.sigma_range);

  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let sx = clamp(i32(x) + dx, 0, i32(params.width) - 1);
      let sy = clamp(i32(y) + dy, 0, i32(params.height) - 1);
      let neighbor = unpack(input[u32(sx) + u32(sy) * params.width]);

      let dist2 = f32(dx * dx + dy * dy);
      let ws = exp(dist2 * inv_2ss);

      let diff = center.xyz - neighbor.xyz;
      let range2 = dot(diff, diff);
      let wr = exp(range2 * inv_2sr);

      let w = ws * wr;
      sum += neighbor * w;
      weight_sum += w;
    }
  }

  let result = vec4<f32>(sum.xyz / weight_sum, center.w);
  output[x + y * params.width] = pack(result);
}
