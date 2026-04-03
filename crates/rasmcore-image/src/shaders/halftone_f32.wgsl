// Halftone — CMYK-style halftone dot pattern, f32 per-pixel
// Converts RGB to CMYK, applies angled sine-wave screen per channel, converts back.

struct Params {
  width: u32,
  height: u32,
  dot_size: f32,
  angle_offset: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const PI: f32 = 3.14159265358979323846;
const DEG2RAD: f32 = PI / 180.0;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = x + y * params.width;
  let px = input[idx];
  let r = px.x;
  let g = px.y;
  let b = px.z;

  // RGB -> CMYK
  let k = 1.0 - max(r, max(g, b));
  var c_val = 0.0;
  var m_val = 0.0;
  var y_val = 0.0;
  if (k < 1.0) {
    let inv_k = 1.0 / (1.0 - k);
    c_val = (1.0 - r - k) * inv_k;
    m_val = (1.0 - g - k) * inv_k;
    y_val = (1.0 - b - k) * inv_k;
  }

  // Standard CMYK screen angles (degrees) + user offset
  let angles = array<f32, 4>(
    (15.0 + params.angle_offset) * DEG2RAD,   // Cyan
    (75.0 + params.angle_offset) * DEG2RAD,   // Magenta
    (0.0 + params.angle_offset) * DEG2RAD,    // Yellow
    (45.0 + params.angle_offset) * DEG2RAD    // Key (Black)
  );
  let cmyk = array<f32, 4>(c_val, m_val, y_val, k);

  let ds = max(params.dot_size, 1.0);
  let freq = PI / ds;
  let xf = f32(x);
  let yf = f32(y);

  var screened: array<f32, 4>;
  for (var i = 0u; i < 4u; i++) {
    let cos_a = cos(angles[i]);
    let sin_a = sin(angles[i]);
    let rx = xf * cos_a + yf * sin_a;
    let ry = -xf * sin_a + yf * cos_a;
    let screen = (sin(rx * freq) * sin(ry * freq) + 1.0) * 0.5;
    if (cmyk[i] > screen) {
      screened[i] = 1.0;
    } else {
      screened[i] = 0.0;
    }
  }

  // CMYK -> RGB: R = (1-C)(1-K), G = (1-M)(1-K), B = (1-Y)(1-K)
  let ro = (1.0 - screened[0]) * (1.0 - screened[3]);
  let go = (1.0 - screened[1]) * (1.0 - screened[3]);
  let bo = (1.0 - screened[2]) * (1.0 - screened[3]);

  output[idx] = vec4<f32>(clamp(ro, 0.0, 1.0), clamp(go, 0.0, 1.0), clamp(bo, 0.0, 1.0), px.w);
}
