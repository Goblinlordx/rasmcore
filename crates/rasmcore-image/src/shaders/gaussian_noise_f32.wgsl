// Gaussian noise — f32 per-pixel with PCG PRNG
// Adds normally-distributed noise using Box-Muller transform

struct Params {
  width: u32,
  height: u32,
  amount: f32,      // noise strength (0-1 normalized from 0-100)
  mean: f32,        // mean offset (normalized from -128..128 to -0.5..0.5)
  sigma: f32,       // std deviation (normalized from 0-100 to 0-0.39)
  seed: u32,        // PRNG seed
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

// PCG hash — deterministic per-pixel pseudo-random
fn pcg(seed: u32) -> u32 {
  var s = seed * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}

// Convert u32 hash to uniform float [0, 1)
fn hash_to_float(h: u32) -> f32 {
  return f32(h) / 4294967296.0;
}

// Box-Muller transform: two uniform randoms → one Gaussian random
fn box_muller(u1: f32, u2: f32) -> f32 {
  let r = sqrt(-2.0 * log(max(u1, 1e-10)));
  return r * cos(6.283185307 * u2);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];

  // Generate per-pixel, per-channel random values
  let base_seed = params.seed ^ (idx * 1664525u + 1013904223u);
  let h0 = pcg(base_seed);
  let h1 = pcg(h0);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let h4 = pcg(h3);
  let h5 = pcg(h4);

  let nr = box_muller(hash_to_float(h0), hash_to_float(h1));
  let ng = box_muller(hash_to_float(h2), hash_to_float(h3));
  let nb = box_muller(hash_to_float(h4), hash_to_float(h5));

  let noise = vec3<f32>(nr, ng, nb) * params.sigma + params.mean;
  let v = clamp(px.rgb + noise * params.amount, vec3<f32>(0.0), vec3<f32>(1.0));
  output[idx] = vec4<f32>(v, px.a);
}
