// Vignette (power-law radial darkening) — f32 per-pixel
// out = pixel * (1 - strength * (dist / max_dist) ^ falloff)
// Supports tile offsets for tiled GPU dispatch

struct Params {
  width: u32,
  height: u32,
  strength: f32,
  falloff: f32,
  full_width: u32,
  full_height: u32,
  offset_x: u32,
  offset_y: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = unpack_f32(input[idx]);

  // Global pixel coordinates (accounting for tile offset)
  let gx = f32(x + params.offset_x);
  let gy = f32(y + params.offset_y);

  // Center of the full image
  let cx = f32(params.full_width) * 0.5;
  let cy = f32(params.full_height) * 0.5;

  // Distance from center, normalised to max possible distance (corner)
  let dx = gx - cx;
  let dy = gy - cy;
  let dist = sqrt(dx * dx + dy * dy);
  let max_dist = sqrt(cx * cx + cy * cy);

  let t = dist / max(max_dist, 1e-6);
  let v = 1.0 - params.strength * pow(t, params.falloff);
  let factor = clamp(v, 0.0, 1.0);

  output[idx] = pack_f32(vec4<f32>(px.rgb * factor, px.a));
}
