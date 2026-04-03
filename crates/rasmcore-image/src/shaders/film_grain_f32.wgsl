// Film grain — spatially correlated noise via block hashing, f32 per-pixel
// amount = grain intensity [0,1], grain_size = block size in pixels, seed = PRNG seed

struct Params {
  width: u32,
  height: u32,
  amount: f32,
  grain_size: f32,
  seed: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
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

// Box-Muller for Gaussian distribution (more film-like than uniform)
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

  // Block coordinates for spatial correlation (coarse grain)
  let bx = u32(floor(f32(x) / params.grain_size));
  let by = u32(floor(f32(y) / params.grain_size));
  let block_seed = params.seed ^ (bx * 1664525u + by * 1013904223u);

  let h0 = pcg(block_seed);
  let h1 = pcg(h0);
  // Single luminance noise value per block (monochromatic grain)
  let grain = box_muller(hash_to_float(h0), hash_to_float(h1)) * 0.15;

  // Luminance-weighted application: more grain in midtones, less in shadows/highlights
  let luma = 0.2126 * px.r + 0.7152 * px.g + 0.0722 * px.b;
  let weight = 4.0 * luma * (1.0 - luma); // peaks at 0.5
  let noise = grain * params.amount * weight;

  let v = clamp(px.rgb + vec3<f32>(noise), vec3<f32>(0.0), vec3<f32>(1.0));
  output[idx] = vec4<f32>(v, px.a);
}
