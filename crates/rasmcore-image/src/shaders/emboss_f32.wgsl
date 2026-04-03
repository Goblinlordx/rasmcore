// Emboss — 3x3 directional emboss kernel convolution, f32 per-pixel
// Kernel: [-2,-1,0; -1,1,1; 0,1,2] (diagonal highlight)

struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn sample_clamped(ix: i32, iy: i32) -> vec4<f32> {
  let sx = u32(clamp(ix, 0, i32(params.width) - 1));
  let sy = u32(clamp(iy, 0, i32(params.height) - 1));
  return input[sx + sy * params.width];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let ix = i32(x);
  let iy = i32(y);

  // Emboss kernel: [-2,-1,0; -1,1,1; 0,1,2]
  var acc = sample_clamped(ix - 1, iy - 1) * (-2.0);
  acc += sample_clamped(ix, iy - 1) * (-1.0);
  // sample_clamped(ix+1, iy-1) * 0.0 — skip
  acc += sample_clamped(ix - 1, iy) * (-1.0);
  acc += sample_clamped(ix, iy) * 1.0;
  acc += sample_clamped(ix + 1, iy) * 1.0;
  // sample_clamped(ix-1, iy+1) * 0.0 — skip
  acc += sample_clamped(ix, iy + 1) * 1.0;
  acc += sample_clamped(ix + 1, iy + 1) * 2.0;

  let alpha = input[x + y * params.width].w;
  let result = vec4<f32>(
    clamp(acc.x, 0.0, 1.0),
    clamp(acc.y, 0.0, 1.0),
    clamp(acc.z, 0.0, 1.0),
    alpha,
  );

  output[x + y * params.width] = result;
}
