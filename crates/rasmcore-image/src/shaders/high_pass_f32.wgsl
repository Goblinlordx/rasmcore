// f32 variant
// High-pass filter — subtract blurred from original, center at 0.5
// This is the SUBTRACTION pass. It expects:
//   - input: the BLURRED image (output of gaussian_blur)
//   - extra_buffers[0]: the ORIGINAL image
// Formula: output[c] = clamp((original[c] - blurred[c]) / 2 + 128, 0, 255)

struct Params {
  width: u32,
  height: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;       // blurred
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> original: array<vec4<f32>>;    // original
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = x + y * params.width;
  let orig = original[idx];
  let blur = input[idx];

  // high_pass = (original - blurred) / 2 + 128, preserve alpha
  let result = vec4<f32>(
    clamp((orig.x - blur.x) * 0.5 + 0.5, 0.0, 1.0),
    clamp((orig.y - blur.y) * 0.5 + 0.5, 0.0, 1.0),
    clamp((orig.z - blur.z) * 0.5 + 0.5, 0.0, 1.0),
    orig.w,
  );

  output[idx] = result;
}
