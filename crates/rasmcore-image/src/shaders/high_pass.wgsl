// High-pass filter — subtract blurred from original, center at 128
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

@group(0) @binding(0) var<storage, read> input: array<u32>;       // blurred
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> original: array<u32>;    // original

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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = x + y * params.width;
  let orig = unpack(original[idx]);
  let blur = unpack(input[idx]);

  // high_pass = (original - blurred) / 2 + 128, preserve alpha
  let result = vec4<f32>(
    clamp((orig.x - blur.x) * 0.5 + 128.0, 0.0, 255.0),
    clamp((orig.y - blur.y) * 0.5 + 128.0, 0.0, 255.0),
    clamp((orig.z - blur.z) * 0.5 + 128.0, 0.0, 255.0),
    orig.w,
  );

  output[idx] = pack(result);
}
