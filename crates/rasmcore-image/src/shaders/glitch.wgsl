// Glitch — horizontal scanline displacement + RGB channel offset

struct Params {
  width: u32,
  height: u32,
  shift_amount: f32,
  channel_offset: f32,
  intensity: f32,
  band_height: u32,
  seed: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Deterministic hash noise — returns value in [-1.0, 1.0]
fn hash_noise(x: u32, s: u32) -> f32 {
  var h = x * 374761393u + s * 1274126177u;
  h = (h ^ (h >> 13u)) * 1103515245u;
  h = h ^ (h >> 16u);
  return f32(h) / f32(0xFFFFFFFFu) * 2.0 - 1.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let band_idx = y / max(params.band_height, 1u);
  let band_noise = hash_noise(band_idx, params.seed);
  let is_glitched = abs(band_noise) < params.intensity;

  let idx = y * params.width + x;

  if (!is_glitched) {
    output[idx] = input[idx];
    return;
  }

  let shift = i32(hash_noise(band_idx, params.seed + 100u) * params.shift_amount);
  let ch_off = i32(params.channel_offset);

  // Sample each channel from shifted position
  let r_sx = clamp(i32(x) + shift + ch_off, 0, i32(params.width) - 1);
  let g_sx = clamp(i32(x) + shift, 0, i32(params.width) - 1);
  let b_sx = clamp(i32(x) + shift - ch_off, 0, i32(params.width) - 1);

  let r_pixel = unpack(input[y * params.width + u32(r_sx)]);
  let g_pixel = unpack(input[y * params.width + u32(g_sx)]);
  let b_pixel = unpack(input[y * params.width + u32(b_sx)]);

  let orig = unpack(input[idx]);
  output[idx] = pack(vec4<f32>(r_pixel.r, g_pixel.g, b_pixel.b, orig.a));
}
