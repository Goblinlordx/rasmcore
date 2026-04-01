// 3D LUT compute shader — applies fused color operations via trilinear interpolation.
//
// The 3D LUT is a cube of RGB output values indexed by input RGB.
// Multiple color adjustments (brightness, contrast, saturation, hue, curves, etc.)
// are pre-baked into the LUT by the CPU, then applied per-pixel on GPU
// in a single pass.
//
// Binding layout:
//   @binding(0) input: array<u32>    — packed RGBA8 pixels (read)
//   @binding(1) output: array<u32>   — packed RGBA8 pixels (write)
//   @binding(2) params: Params       — width, height, lut_size
//   @binding(3) lut: array<vec4<f32>> — 3D LUT as flat array (size^3 entries, RGBA float)

struct Params {
  width: u32,
  height: u32,
  lut_size: u32,  // typically 33 or 65
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> lut: array<f32>;

fn unpack(pixel: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(pixel & 0xFFu),
    f32((pixel >> 8u) & 0xFFu),
    f32((pixel >> 16u) & 0xFFu),
    f32((pixel >> 24u) & 0xFFu),
  );
}

fn pack(color: vec4<f32>) -> u32 {
  let r = u32(clamp(color.x, 0.0, 255.0));
  let g = u32(clamp(color.y, 0.0, 255.0));
  let b = u32(clamp(color.z, 0.0, 255.0));
  let a = u32(clamp(color.w, 0.0, 255.0));
  return r | (g << 8u) | (b << 16u) | (a << 24u);
}

// Read RGBA from the flat LUT array at index (r, g, b)
fn lut_sample(ri: u32, gi: u32, bi: u32) -> vec4<f32> {
  let size = params.lut_size;
  let idx = (bi * size * size + gi * size + ri) * 4u;
  return vec4<f32>(lut[idx], lut[idx + 1u], lut[idx + 2u], lut[idx + 3u]);
}

// Trilinear interpolation in the 3D LUT
fn lut_lookup(r: f32, g: f32, b: f32) -> vec3<f32> {
  let size_f = f32(params.lut_size - 1u);

  // Scale [0..255] -> [0..size-1]
  let rf = clamp(r / 255.0 * size_f, 0.0, size_f);
  let gf = clamp(g / 255.0 * size_f, 0.0, size_f);
  let bf = clamp(b / 255.0 * size_f, 0.0, size_f);

  // Integer and fractional parts
  let r0 = u32(floor(rf));
  let g0 = u32(floor(gf));
  let b0 = u32(floor(bf));
  let r1 = min(r0 + 1u, params.lut_size - 1u);
  let g1 = min(g0 + 1u, params.lut_size - 1u);
  let b1 = min(b0 + 1u, params.lut_size - 1u);
  let fr = rf - floor(rf);
  let fg = gf - floor(gf);
  let fb = bf - floor(bf);

  // 8 corner samples
  let c000 = lut_sample(r0, g0, b0).xyz;
  let c100 = lut_sample(r1, g0, b0).xyz;
  let c010 = lut_sample(r0, g1, b0).xyz;
  let c110 = lut_sample(r1, g1, b0).xyz;
  let c001 = lut_sample(r0, g0, b1).xyz;
  let c101 = lut_sample(r1, g0, b1).xyz;
  let c011 = lut_sample(r0, g1, b1).xyz;
  let c111 = lut_sample(r1, g1, b1).xyz;

  // Trilinear interpolation
  let c00 = mix(c000, c100, fr);
  let c10 = mix(c010, c110, fr);
  let c01 = mix(c001, c101, fr);
  let c11 = mix(c011, c111, fr);
  let c0 = mix(c00, c10, fg);
  let c1 = mix(c01, c11, fg);
  return mix(c0, c1, fb);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = x + y * params.width;
  let pixel = unpack(input[idx]);

  // Apply 3D LUT — preserves alpha
  let mapped = lut_lookup(pixel.x, pixel.y, pixel.z);
  output[idx] = pack(vec4<f32>(mapped, pixel.w));
}
