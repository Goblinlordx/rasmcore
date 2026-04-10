//! ThresholdBinary filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::helpers::luminance;
/// Binary threshold — pixels above threshold become white, below become black.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "threshold_binary", category = "adjustment", cost = "O(n)")]
pub struct ThresholdBinary {
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.5)]
    pub threshold: f32,
}

impl Filter for ThresholdBinary {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let t = self.threshold;
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            let l = luminance(px[0], px[1], px[2]);
            let v = if l >= t { 1.0 } else { 0.0 };
            px[0] = v;
            px[1] = v;
            px[2] = v;
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(THRESHOLD_WGSL)
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.threshold.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        Some(buf)
    }
}

const THRESHOLD_WGSL: &str = r#"
struct Params { width: u32, height: u32, threshold: f32, _pad: u32, }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    let l = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
    let v = select(0.0, 1.0, l >= params.threshold);
    store_pixel(idx, vec4(v, v, v, p.w));
}
"#;
