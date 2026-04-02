// ── Format-agnostic buffer I/O — f32 vec4 variant ─────────────────────────────
// Declares input/output bindings as array<vec4<f32>> and provides load_pixel/
// store_pixel as identity operations. Same API as io_u32.wgsl — shader bodies
// are format-agnostic and compose with either variant.

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;

fn load_pixel(idx: u32) -> vec4<f32> {
  return input[idx];
}

fn store_pixel(idx: u32, color: vec4<f32>) {
  output[idx] = color;
}
