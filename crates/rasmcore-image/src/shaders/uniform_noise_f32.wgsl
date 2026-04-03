// Uniform noise — f32 per-pixel with PCG PRNG
// Adds uniformly distributed noise in [-range, +range]

struct Params {
  width: u32,
  height: u32,
  range_norm: f32,  // range / 255.0
  seed: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn pcg(seed: u32) -> u32 {
  var s = seed * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hash_to_float(h: u32) -> f32 {
  return f32(h) / 4294967296.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];

  let base_seed = params.seed ^ (idx * 1664525u + 1013904223u);
  let h0 = pcg(base_seed);
  let h1 = pcg(h0);
  let h2 = pcg(h1);

  // Uniform in [-range, +range]
  let nr = (hash_to_float(h0) * 2.0 - 1.0) * params.range_norm;
  let ng = (hash_to_float(h1) * 2.0 - 1.0) * params.range_norm;
  let nb = (hash_to_float(h2) * 2.0 - 1.0) * params.range_norm;

  let v = clamp(px.rgb + vec3<f32>(nr, ng, nb), vec3<f32>(0.0), vec3<f32>(1.0));
  output[idx] = vec4<f32>(v, px.a);
}
