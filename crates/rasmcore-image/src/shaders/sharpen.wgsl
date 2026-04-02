// Unsharp mask sharpen — sharpen = original + amount * (original - blur)
// Single pass: computes inline 3x3 Gaussian blur and applies unsharp mask.

struct Params {
  width: u32,
  height: u32,
  amount: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
fn sample_clamped(ix: i32, iy: i32) -> vec4<f32> {
  let sx = u32(clamp(ix, 0, i32(params.width) - 1));
  let sy = u32(clamp(iy, 0, i32(params.height) - 1));
  return unpack(input[sx + sy * params.width]);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let ix = i32(x);
  let iy = i32(y);
  let original = unpack(input[x + y * params.width]);

  // 3x3 Gaussian blur approximation (1/16 kernel: 1 2 1 / 2 4 2 / 1 2 1)
  var blurred = sample_clamped(ix - 1, iy - 1) * (1.0 / 16.0);
  blurred += sample_clamped(ix, iy - 1) * (2.0 / 16.0);
  blurred += sample_clamped(ix + 1, iy - 1) * (1.0 / 16.0);
  blurred += sample_clamped(ix - 1, iy) * (2.0 / 16.0);
  blurred += original * (4.0 / 16.0);
  blurred += sample_clamped(ix + 1, iy) * (2.0 / 16.0);
  blurred += sample_clamped(ix - 1, iy + 1) * (1.0 / 16.0);
  blurred += sample_clamped(ix, iy + 1) * (2.0 / 16.0);
  blurred += sample_clamped(ix + 1, iy + 1) * (1.0 / 16.0);

  // Unsharp mask: result = original + amount * (original - blurred)
  let detail = original - blurred;
  let result = vec4<f32>(
    clamp(original.x + params.amount * detail.x, 0.0, 255.0),
    clamp(original.y + params.amount * detail.y, 0.0, 255.0),
    clamp(original.z + params.amount * detail.z, 0.0, 255.0),
    original.w,
  );

  output[x + y * params.width] = pack(result);
}
