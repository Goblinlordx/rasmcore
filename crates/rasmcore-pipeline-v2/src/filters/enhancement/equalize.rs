use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

/// Histogram equalization — maximizes contrast via CDF remapping.
///
/// Quantizes to 256 bins for CDF computation, then remaps.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "equalize", category = "enhancement", cost = "O(n)")]
pub struct Equalize;

impl Filter for Equalize {
    fn analysis_buffer_outputs(&self) -> &'static [crate::analysis_buffer::AnalysisBufferDecl] {
        use crate::analysis_buffer::{AnalysisBufferDecl, AnalysisBufferKind};
        static DECLS: [AnalysisBufferDecl; 1] = [AnalysisBufferDecl {
            logical_id: 0,
            kind: AnalysisBufferKind::Histogram256,
            size_bytes: 0,
        }];
        &DECLS
    }

    fn gpu_shader_passes_with_context(
        &self,
        width: u32,
        height: u32,
        mapping: &crate::analysis_buffer::NodeBufferMapping,
    ) -> Option<Vec<crate::node::GpuShader>> {
        use crate::gpu_shaders::reduction::GpuReduction;
        let resolved_id = mapping.resolve(0);
        let reduction = GpuReduction::histogram_256(256).with_buffer_id(resolved_id);
        let passes = reduction.build_passes(width, height);
        Some(vec![passes.pass1, passes.pass2])
    }

    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let npixels = input.len() / 4;
        let mut out = input.to_vec();

        // Per-channel histogram equalization
        for c in 0..3 {
            let mut hist = [0u32; 256];
            for pixel in input.chunks_exact(4) {
                let bin = ((pixel[c].max(0.0) * 255.0) as usize).min(255);
                hist[bin] += 1;
            }

            // Build CDF
            let mut cdf = [0u32; 256];
            cdf[0] = hist[0];
            for i in 1..256 {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
            let denom = (npixels as u32).saturating_sub(cdf_min);

            if denom > 0 {
                for pixel in out.chunks_exact_mut(4) {
                    let bin = ((pixel[c].max(0.0) * 255.0) as usize).min(255);
                    pixel[c] = (cdf[bin] - cdf_min) as f32 / denom as f32;
                }
            }
        }

        Ok(out)
    }
}

// ── Equalize GPU (3-pass via Histogram256 reduction) ───────────────────────

/// Equalize apply shader — reads per-channel histogram, computes CDF inline, remaps.
const EQUALIZE_APPLY_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  total_pixels: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;

// Compute CDF value for a given bin and channel offset
fn cdf_at(bin: u32, channel_offset: u32) -> f32 {
  var sum = 0u;
  for (var i = 0u; i <= bin; i = i + 1u) {
    sum += histogram[channel_offset + i];
  }
  // Find cdf_min (first non-zero bin)
  var cdf_min = 0u;
  for (var i = 0u; i < 256u; i = i + 1u) {
    let v = histogram[channel_offset + i];
    if (v > 0u) {
      cdf_min = v;
      // Compute cdf_min as cumulative at first non-zero
      var s = 0u;
      for (var j = 0u; j <= i; j = j + 1u) {
        s += histogram[channel_offset + j];
      }
      cdf_min = s;
      break;
    }
  }
  let denom = params.total_pixels - cdf_min;
  if (denom == 0u) { return f32(bin) / 255.0; }
  return f32(sum - cdf_min) / f32(denom);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let pixel = input[idx];
  let bin_r = u32(clamp(pixel.x * 255.0, 0.0, 255.0));
  let bin_g = u32(clamp(pixel.y * 255.0, 0.0, 255.0));
  let bin_b = u32(clamp(pixel.z * 255.0, 0.0, 255.0));

  output[idx] = vec4<f32>(
    cdf_at(bin_r, 0u),
    cdf_at(bin_g, 256u),
    cdf_at(bin_b, 512u),
    pixel.w,
  );
}
"#;

impl GpuFilter for Equalize {
    fn shader_body(&self) -> &str {
        EQUALIZE_APPLY_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [256, 1, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let total = width * height;
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&total.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn gpu_shaders(&self, width: u32, height: u32) -> Vec<crate::node::GpuShader> {
        use crate::gpu_shaders::reduction::GpuReduction;

        let reduction = GpuReduction::histogram_256(256);
        let passes = reduction.build_passes(width, height);

        let pass3 = crate::node::GpuShader::new(
            EQUALIZE_APPLY_WGSL.to_string(),
            "main",
            [256, 1, 1],
            self.params(width, height),
        )
        .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);

        vec![passes.pass1, passes.pass2, pass3]
    }
}
