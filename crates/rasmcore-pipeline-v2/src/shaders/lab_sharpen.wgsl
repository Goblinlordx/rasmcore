// Lab sharpen — convert to Lab, sharpen L channel only, convert back.
// Single-pass approximation: inline Lab conversion + unsharp mask on L.

struct Params {
  width: u32,
  height: u32,
  amount: f32,
  radius: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

// Simplified sRGB→Lab L channel (perceptual lightness)
fn linear_to_L(r: f32, g: f32, b: f32) -> f32 {
  // Y (CIE luminance) from linear sRGB
  let Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  // CIE L* from Y (D65 illuminant, Yn = 1.0)
  let eps = 0.008856;
  let kappa = 903.3;
  if (Y > eps) {
    return 116.0 * pow(Y, 1.0 / 3.0) - 16.0;
  } else {
    return kappa * Y;
  }
}

fn clamp_coord(v: i32, size: u32) -> u32 {
  return u32(clamp(v, 0, i32(size) - 1));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let r = i32(params.radius);
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);

  // Compute L for center pixel
  let L_center = linear_to_L(pixel.x, pixel.y, pixel.z);

  // Compute average L in neighborhood (box blur approximation)
  var L_sum: f32 = 0.0;
  var count: f32 = 0.0;
  for (var dy: i32 = -r; dy <= r; dy++) {
    for (var dx: i32 = -r; dx <= r; dx++) {
      let sx = clamp_coord(x + dx, params.width);
      let sy = clamp_coord(y + dy, params.height);
      let p = load_pixel(sy * params.width + sx);
      L_sum += linear_to_L(p.x, p.y, p.z);
      count += 1.0;
    }
  }
  let L_blur = L_sum / count;

  // Unsharp mask on L channel
  let L_detail = L_center - L_blur;
  let L_sharp = L_center + L_detail * params.amount;

  // Apply L ratio to RGB (preserves chrominance)
  let ratio = select(1.0, L_sharp / L_center, L_center > 0.001);
  let out_r = clamp(pixel.x * ratio, 0.0, 100.0);
  let out_g = clamp(pixel.y * ratio, 0.0, 100.0);
  let out_b = clamp(pixel.z * ratio, 0.0, 100.0);

  store_pixel(idx, vec4<f32>(out_r, out_g, out_b, pixel.w));
}
