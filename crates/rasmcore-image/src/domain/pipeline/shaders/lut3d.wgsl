// 3D color LUT shader — trilinear interpolation
// Applies a pre-fused 3D color lookup table to RGBA8 pixel data.
// The LUT encapsulates N fused color operations in a single pass.

struct Params {
  width: u32,
  height: u32,
  grid_size: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
// LUT data: grid^3 entries, each entry is vec4<f32> (rgb + padding)
@group(0) @binding(3) var<storage, read> lut: array<vec4<f32>>;
// Index into 3D LUT: lut[b * grid^2 + g * grid + r]
fn lut_index(r: u32, g: u32, b: u32, grid: u32) -> u32 {
  return b * grid * grid + g * grid + r;
}

// Trilinear interpolation in the 3D LUT
fn lut_lookup(rf: f32, gf: f32, bf: f32, grid: u32) -> vec3<f32> {
  let gs = f32(grid - 1u);

  // Scale to grid coordinates
  let r = clamp(rf * gs, 0.0, gs);
  let g = clamp(gf * gs, 0.0, gs);
  let b = clamp(bf * gs, 0.0, gs);

  // Floor and ceil indices
  let r0 = u32(floor(r)); let r1 = min(r0 + 1u, grid - 1u);
  let g0 = u32(floor(g)); let g1 = min(g0 + 1u, grid - 1u);
  let b0 = u32(floor(b)); let b1 = min(b0 + 1u, grid - 1u);

  // Fractional parts
  let fr = r - floor(r);
  let fg = g - floor(g);
  let fb = b - floor(b);

  // 8 corner lookups
  let c000 = lut[lut_index(r0, g0, b0, grid)].xyz;
  let c100 = lut[lut_index(r1, g0, b0, grid)].xyz;
  let c010 = lut[lut_index(r0, g1, b0, grid)].xyz;
  let c110 = lut[lut_index(r1, g1, b0, grid)].xyz;
  let c001 = lut[lut_index(r0, g0, b1, grid)].xyz;
  let c101 = lut[lut_index(r1, g0, b1, grid)].xyz;
  let c011 = lut[lut_index(r0, g1, b1, grid)].xyz;
  let c111 = lut[lut_index(r1, g1, b1, grid)].xyz;

  // Trilinear interpolation
  let c00 = mix(c000, c100, fr);
  let c10 = mix(c010, c110, fr);
  let c01 = mix(c001, c101, fr);
  let c11 = mix(c011, c111, fr);
  let c0 = mix(c00, c10, fg);
  let c1 = mix(c01, c11, fg);
  return mix(c0, c1, fb);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.width + gid.x;
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  let pixel = unpack(input[idx]);

  // Normalize to [0, 1]
  let rf = pixel.x / 255.0;
  let gf = pixel.y / 255.0;
  let bf = pixel.z / 255.0;

  // Apply 3D LUT
  let result = lut_lookup(rf, gf, bf, params.grid_size);

  // Output: LUT returns [0, 1], scale back to [0, 255]
  let out_color = vec4<f32>(
    result.x * 255.0,
    result.y * 255.0,
    result.z * 255.0,
    pixel.w,  // preserve alpha
  );

  output[idx] = pack(out_color);
}
