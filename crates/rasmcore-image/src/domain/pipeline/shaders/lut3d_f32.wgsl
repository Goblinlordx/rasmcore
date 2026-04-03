// 3D color LUT shader — trilinear interpolation, f32 I/O
// Input/output: vec4<f32> in [0,1]. LUT data: vec4<f32> (rgb + padding).

struct Params {
  width: u32,
  height: u32,
  grid_size: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> lut: array<vec4<f32>>;

fn lut_index(r: u32, g: u32, b: u32, grid: u32) -> u32 {
  return b * grid * grid + g * grid + r;
}

fn lut_lookup(rf: f32, gf: f32, bf: f32, grid: u32) -> vec3<f32> {
  let gs = f32(grid - 1u);
  let r = clamp(rf * gs, 0.0, gs);
  let g = clamp(gf * gs, 0.0, gs);
  let b = clamp(bf * gs, 0.0, gs);

  let r0 = u32(floor(r)); let r1 = min(r0 + 1u, grid - 1u);
  let g0 = u32(floor(g)); let g1 = min(g0 + 1u, grid - 1u);
  let b0 = u32(floor(b)); let b1 = min(b0 + 1u, grid - 1u);

  let fr = r - floor(r);
  let fg = g - floor(g);
  let fb = b - floor(b);

  let c000 = lut[lut_index(r0, g0, b0, grid)].xyz;
  let c100 = lut[lut_index(r1, g0, b0, grid)].xyz;
  let c010 = lut[lut_index(r0, g1, b0, grid)].xyz;
  let c110 = lut[lut_index(r1, g1, b0, grid)].xyz;
  let c001 = lut[lut_index(r0, g0, b1, grid)].xyz;
  let c101 = lut[lut_index(r1, g0, b1, grid)].xyz;
  let c011 = lut[lut_index(r0, g1, b1, grid)].xyz;
  let c111 = lut[lut_index(r1, g1, b1, grid)].xyz;

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
  let px = input[idx];

  let result = lut_lookup(px.r, px.g, px.b, params.grid_size);
  output[idx] = vec4<f32>(result, px.a);
}
