// Ordered dither (Bayer 8x8) — GPU-friendly alternative to Floyd-Steinberg.
// Floyd-Steinberg requires sequential error propagation (not parallelizable).
// Bayer dither is fully parallel with visually similar results.

struct Params {
  width: u32,
  height: u32,
  levels: f32,  // 1.0/max_colors quantization step
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

// Bayer 8x8 threshold matrix (normalized to [0, 1))
const BAYER8: array<f32, 64> = array<f32, 64>(
   0.0/64.0, 48.0/64.0, 12.0/64.0, 60.0/64.0,  3.0/64.0, 51.0/64.0, 15.0/64.0, 63.0/64.0,
  32.0/64.0, 16.0/64.0, 44.0/64.0, 28.0/64.0, 35.0/64.0, 19.0/64.0, 47.0/64.0, 31.0/64.0,
   8.0/64.0, 56.0/64.0,  4.0/64.0, 52.0/64.0, 11.0/64.0, 59.0/64.0,  7.0/64.0, 55.0/64.0,
  40.0/64.0, 24.0/64.0, 36.0/64.0, 20.0/64.0, 43.0/64.0, 27.0/64.0, 39.0/64.0, 23.0/64.0,
   2.0/64.0, 50.0/64.0, 14.0/64.0, 62.0/64.0,  1.0/64.0, 49.0/64.0, 13.0/64.0, 61.0/64.0,
  34.0/64.0, 18.0/64.0, 46.0/64.0, 30.0/64.0, 33.0/64.0, 17.0/64.0, 45.0/64.0, 29.0/64.0,
  10.0/64.0, 58.0/64.0,  6.0/64.0, 54.0/64.0,  9.0/64.0, 57.0/64.0,  5.0/64.0, 53.0/64.0,
  42.0/64.0, 26.0/64.0, 38.0/64.0, 22.0/64.0, 41.0/64.0, 25.0/64.0, 37.0/64.0, 21.0/64.0,
);

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);

  // Bayer threshold for this pixel position
  let bx = gid.x % 8u;
  let by = gid.y % 8u;
  let threshold = BAYER8[by * 8u + bx] - 0.5; // center around 0

  // Quantize each channel with dither offset
  let step = params.levels;
  let r = floor((pixel.x + threshold * step) / step + 0.5) * step;
  let g = floor((pixel.y + threshold * step) / step + 0.5) * step;
  let b = floor((pixel.z + threshold * step) / step + 0.5) * step;

  store_pixel(idx, vec4<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), pixel.w));
}
