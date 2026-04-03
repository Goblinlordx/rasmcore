// K-means quantize — nearest-color palette lookup.
// Palette is pre-computed on CPU and passed as extra buffer.
// GPU does per-pixel nearest-neighbor search in the palette.

struct Params {
  width: u32,
  height: u32,
  k: u32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> palette: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);

  // Find nearest palette color (Euclidean distance in RGB)
  var best_dist: f32 = 1e10;
  var best_color = vec4<f32>(0.0, 0.0, 0.0, pixel.w);
  for (var i: u32 = 0u; i < params.k; i++) {
    let pal = palette[i];
    let dr = pixel.x - pal.x;
    let dg = pixel.y - pal.y;
    let db = pixel.z - pal.z;
    let dist = dr * dr + dg * dg + db * db;
    if (dist < best_dist) {
      best_dist = dist;
      best_color = vec4<f32>(pal.x, pal.y, pal.z, pixel.w);
    }
  }

  store_pixel(idx, best_color);
}
