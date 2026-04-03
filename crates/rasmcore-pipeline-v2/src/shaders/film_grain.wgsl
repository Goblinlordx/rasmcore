// Film grain — noise overlay weighted by midtone luminance.
// Requires noise_2d() from NOISE_WGSL (composed at runtime).

struct Params {
  width: u32,
  height: u32,
  amount: f32,
  inv_size: f32,
  seed_lo: u32,
  seed_hi: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);

  // Grain at reduced resolution (size > 1 = coarser grain)
  let gx = u32(f32(gid.x) * params.inv_size);
  let gy = u32(f32(gid.y) * params.inv_size);
  let grain = noise_2d(gx, gy, params.seed_lo, params.seed_hi) * params.amount;

  // Weight by midtone luminance (more visible in midtones)
  let luma = clamp(0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z, 0.0, 1.0);
  let weight = 4.0 * luma * (1.0 - luma);
  let g = grain * weight;

  store_pixel(idx, vec4<f32>(pixel.x + g, pixel.y + g, pixel.z + g, pixel.w));
}
