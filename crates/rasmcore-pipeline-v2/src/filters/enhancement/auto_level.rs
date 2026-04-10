use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

/// Auto-level — linear stretch from actual min to actual max.
///
/// Finds per-channel min/max across all pixels and linearly maps to [0, 1].
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "auto_level", category = "enhancement", cost = "O(n)")]
pub struct AutoLevel;

impl Filter for AutoLevel {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for pixel in input.chunks_exact(4) {
            min[0] = min[0].min(pixel[0]);
            min[1] = min[1].min(pixel[1]);
            min[2] = min[2].min(pixel[2]);
            max[0] = max[0].max(pixel[0]);
            max[1] = max[1].max(pixel[1]);
            max[2] = max[2].max(pixel[2]);
        }

        let range = [
            (max[0] - min[0]).max(1e-10),
            (max[1] - min[1]).max(1e-10),
            (max[2] - min[2]).max(1e-10),
        ];
        let inv_range = [1.0 / range[0], 1.0 / range[1], 1.0 / range[2]];
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = (pixel[0] - min[0]) * inv_range[0];
            pixel[1] = (pixel[1] - min[1]) * inv_range[1];
            pixel[2] = (pixel[2] - min[2]) * inv_range[2];
        }
        Ok(out)
    }
}

// ── AutoLevel GPU (3-pass via ChannelMinMax reduction) ──────────────────────

/// AutoLevel apply shader — reads min/max from reduction buffer, stretches per-channel.
const AUTO_LEVEL_APPLY_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> reduction: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let ch_min = reduction[0].xyz;
  let ch_max = reduction[1].xyz;
  let range = max(ch_max - ch_min, vec3<f32>(0.00001, 0.00001, 0.00001));

  let pixel = input[idx];
  output[idx] = vec4<f32>(
    (pixel.x - ch_min.x) / range.x,
    (pixel.y - ch_min.y) / range.y,
    (pixel.z - ch_min.z) / range.z,
    pixel.w,
  );
}
"#;

impl GpuFilter for AutoLevel {
    fn shader_body(&self) -> &str {
        AUTO_LEVEL_APPLY_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [256, 1, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn gpu_shaders(&self, width: u32, height: u32) -> Vec<crate::node::GpuShader> {
        use crate::gpu_shaders::reduction::GpuReduction;

        let reduction = GpuReduction::channel_min_max(256);
        let passes = reduction.build_passes(width, height);

        let pass3 = crate::node::GpuShader::new(
            AUTO_LEVEL_APPLY_WGSL.to_string(),
            "main",
            [256, 1, 1],
            self.params(width, height),
        )
        .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);

        vec![passes.pass1, passes.pass2, pass3]
    }
}
