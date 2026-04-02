// ── Bilinear sampling from input buffer ──────────────────────────────────────
// REQUIRES: fn unpack() is defined (include pixel_ops first)
// REQUIRES: Params struct has width and height as first two u32 fields
// REQUIRES: @group(0) @binding(0) var<storage, read> input: array<u32>

fn sample_bilinear(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let dx = fx - f32(ix);
  let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.width) - 1);
  let y0 = clamp(iy, 0, i32(params.height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.height) - 1);
  let p00 = unpack(input[u32(x0) + u32(y0) * params.width]);
  let p10 = unpack(input[u32(x1) + u32(y0) * params.width]);
  let p01 = unpack(input[u32(x0) + u32(y1) * params.width]);
  let p11 = unpack(input[u32(x1) + u32(y1) * params.width]);
  return mix(mix(p00, p10, dx), mix(p01, p11, dx), dy);
}
