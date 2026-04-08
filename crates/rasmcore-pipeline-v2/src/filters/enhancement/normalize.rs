use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

/// Normalize — linear contrast stretch with 2% black / 1% white clipping.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "normalize", category = "enhancement", cost = "O(n)")]
pub struct Normalize {
    #[param(min = 0.0, max = 0.5, default = 0.02)]
    pub black_clip: f32,
    #[param(min = 0.0, max = 0.5, default = 0.01)]
    pub white_clip: f32,
}

impl Filter for Normalize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let npixels = input.len() / 4;
        let mut out = input.to_vec();

        for c in 0..3 {
            let mut hist = [0u32; 256];
            for pixel in input.chunks_exact(4) {
                let bin = ((pixel[c].max(0.0) * 255.0) as usize).min(255);
                hist[bin] += 1;
            }

            // Find black point (skip bottom black_clip fraction)
            let black_threshold = (npixels as f32 * self.black_clip) as u32;
            let mut accum = 0u32;
            let mut black_bin = 0;
            for (i, &h) in hist.iter().enumerate() {
                accum += h;
                if accum >= black_threshold {
                    black_bin = i;
                    break;
                }
            }

            // Find white point (skip top white_clip fraction)
            let white_threshold = (npixels as f32 * self.white_clip) as u32;
            accum = 0;
            let mut white_bin = 255;
            for i in (0..256).rev() {
                accum += hist[i];
                if accum >= white_threshold {
                    white_bin = i;
                    break;
                }
            }

            let black = black_bin as f32 / 255.0;
            let white = white_bin as f32 / 255.0;
            let range = white - black;

            if range > 1e-10 {
                for pixel in out.chunks_exact_mut(4) {
                    pixel[c] = (pixel[c] - black) / range;
                }
            }
        }

        Ok(out)
    }
}

// ── Normalize GPU (3-pass via Histogram256 reduction) ──────────────────────

/// Normalize apply shader — reads histogram, finds percentile clip points, stretches.
const NORMALIZE_APPLY_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  total_pixels: u32,
  black_clip_count: u32,
  white_clip_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;

// Find the black point (percentile clip from bottom) for a channel
fn find_black(channel_offset: u32) -> f32 {
  var accum = 0u;
  for (var i = 0u; i < 256u; i = i + 1u) {
    accum += histogram[channel_offset + i];
    if (accum >= params.black_clip_count) {
      return f32(i) / 255.0;
    }
  }
  return 0.0;
}

// Find the white point (percentile clip from top) for a channel
fn find_white(channel_offset: u32) -> f32 {
  var accum = 0u;
  for (var i = 255u; ; i = i - 1u) {
    accum += histogram[channel_offset + i];
    if (accum >= params.white_clip_count) {
      return f32(i) / 255.0;
    }
    if (i == 0u) { break; }
  }
  return 1.0;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let black_r = find_black(0u);
  let black_g = find_black(256u);
  let black_b = find_black(512u);
  let white_r = find_white(0u);
  let white_g = find_white(256u);
  let white_b = find_white(512u);

  let range_r = max(white_r - black_r, 0.00001);
  let range_g = max(white_g - black_g, 0.00001);
  let range_b = max(white_b - black_b, 0.00001);

  let pixel = input[idx];
  output[idx] = vec4<f32>(
    (pixel.x - black_r) / range_r,
    (pixel.y - black_g) / range_g,
    (pixel.z - black_b) / range_b,
    pixel.w,
  );
}
"#;

impl GpuFilter for Normalize {
    fn shader_body(&self) -> &str { NORMALIZE_APPLY_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let total = width * height;
        let black_clip_count = (total as f32 * self.black_clip) as u32;
        let white_clip_count = (total as f32 * self.white_clip) as u32;
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&total.to_le_bytes());
        buf.extend_from_slice(&black_clip_count.to_le_bytes());
        buf.extend_from_slice(&white_clip_count.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn gpu_shaders(&self, width: u32, height: u32) -> Vec<crate::node::GpuShader> {
        use crate::gpu_shaders::reduction::GpuReduction;

        let reduction = GpuReduction::histogram_256(256);
        let passes = reduction.build_passes(width, height);

        let pass3 = crate::node::GpuShader::new(
            NORMALIZE_APPLY_WGSL.to_string(),
            "main",
            [256, 1, 1],
            self.params(width, height),
        )
        .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);

        vec![passes.pass1, passes.pass2, pass3]
    }
}
