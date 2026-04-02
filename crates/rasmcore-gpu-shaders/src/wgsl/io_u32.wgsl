// ── Format-agnostic buffer I/O — u32-packed RGBA8 variant ─────────────────────
// Declares input/output bindings as array<u32> and provides load_pixel/store_pixel
// that convert to/from vec4<f32> via pack/unpack. Shader bodies use only these
// abstract functions and never reference the buffer type directly.

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn load_pixel(idx: u32) -> vec4<f32> {
  let pixel = input[idx];
  return vec4<f32>(
    f32(pixel & 0xFFu),
    f32((pixel >> 8u) & 0xFFu),
    f32((pixel >> 16u) & 0xFFu),
    f32((pixel >> 24u) & 0xFFu),
  );
}

fn store_pixel(idx: u32, color: vec4<f32>) {
  let r = u32(clamp(color.x, 0.0, 255.0));
  let g = u32(clamp(color.y, 0.0, 255.0));
  let b = u32(clamp(color.z, 0.0, 255.0));
  let a = u32(clamp(color.w, 0.0, 255.0));
  output[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
