// Posterize — reduce color levels, f32 per-pixel
// levels = number of distinct levels per channel (2-256)

struct Params {
  width: u32,
  height: u32,
  levels: f32,
  _pad: u32,
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
  let px = input[idx];
  let levels = max(params.levels, 2.0);
  let v = floor(px.rgb * levels) / (levels - 1.0);
  output[idx] = vec4<f32>(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)), px.a);
}
