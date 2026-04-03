// Generic NxN convolution — f32 variant
// Applies a kw x kh kernel to each pixel with clamped border handling.
// Kernel weights are pre-divided by divisor on the host side.

struct Params {
  width: u32,
  height: u32,
  kw: u32,
  kh: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

fn sample_clamped(ix: i32, iy: i32) -> vec4<f32> {
  let sx = u32(clamp(ix, 0, i32(params.width) - 1));
  let sy = u32(clamp(iy, 0, i32(params.height) - 1));
  return unpack_f32(input[sx + sy * params.width]);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let ix = i32(x);
  let iy = i32(y);
  let half_w = i32(params.kw) / 2;
  let half_h = i32(params.kh) / 2;

  var sum = vec4<f32>(0.0);
  for (var ky = 0u; ky < params.kh; ky++) {
    let sy = iy - half_h + i32(ky);
    for (var kx = 0u; kx < params.kw; kx++) {
      let sx = ix - half_w + i32(kx);
      sum += sample_clamped(sx, sy) * kernel[ky * params.kw + kx];
    }
  }

  output[x + y * params.width] = pack_f32(clamp(sum, vec4<f32>(0.0), vec4<f32>(1.0)));
}
