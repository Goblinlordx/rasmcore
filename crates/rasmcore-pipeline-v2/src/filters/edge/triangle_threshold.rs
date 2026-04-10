//! TriangleThreshold filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::luminance;

/// Triangle threshold — automatic threshold for unimodal histograms.
/// Finds the threshold that maximizes the distance from the histogram
/// line connecting the peak to the farthest bin.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "triangle_threshold", category = "adjustment", cost = "O(n)")]
pub struct TriangleThreshold;

impl Filter for TriangleThreshold {
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

    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let mut hist = [0u32; 256];
        for px in input.chunks_exact(4) {
            let l = luminance(px[0], px[1], px[2]);
            let bin = ((l * 255.0).round().max(0.0) as usize).min(255);
            hist[bin] += 1;
        }
        // Find peak
        let peak_idx = hist
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Find farthest non-zero bin from peak
        let far_idx = if peak_idx < 128 {
            hist.iter().rposition(|&h| h > 0).unwrap_or(255)
        } else {
            hist.iter().position(|&h| h > 0).unwrap_or(0)
        };
        // Line from (peak_idx, hist[peak]) to (far_idx, hist[far_idx])
        let (x1, y1) = (peak_idx as f64, hist[peak_idx] as f64);
        let (x2, y2) = (far_idx as f64, hist[far_idx] as f64);
        let len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt().max(1e-6);
        // Find bin with max distance from line
        let mut best_t = peak_idx;
        let mut max_dist = 0.0f64;
        let (lo, hi) = if peak_idx < far_idx {
            (peak_idx, far_idx)
        } else {
            (far_idx, peak_idx)
        };
        for t in lo..=hi {
            let d =
                ((y2 - y1) * t as f64 - (x2 - x1) * hist[t] as f64 + x2 * y1 - y2 * x1).abs() / len;
            if d > max_dist {
                max_dist = d;
                best_t = t;
            }
        }
        let threshold = best_t as f32 / 255.0;
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            let l = luminance(px[0], px[1], px[2]);
            let v = if l >= threshold { 1.0 } else { 0.0 };
            px[0] = v;
            px[1] = v;
            px[2] = v;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        use crate::gpu_shaders::reduction::GpuReduction;
        let reduction = GpuReduction::histogram_256(256);
        let passes = reduction.build_passes(width, height);
        let total = width * height;
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&total.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        let pass3 = GpuShader::new(TRIANGLE_APPLY_WGSL.to_string(), "main", [256, 1, 1], params)
            .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);
        Some(vec![passes.pass1, passes.pass2, pass3])
    }
}

/// Triangle threshold GPU apply — reads histogram, finds peak/far, computes threshold.
const TRIANGLE_APPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, total_pixels: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_pixels) { return; }

    // Find peak bin in luminance histogram
    var peak_idx = 0u;
    var peak_val = 0u;
    for (var i = 0u; i < 256u; i++) {
        if (histogram[i] > peak_val) { peak_val = histogram[i]; peak_idx = i; }
    }
    // Find farthest non-zero bin
    var far_idx = 0u;
    if (peak_idx < 128u) {
        for (var i = 255u; i > 0u; i--) { if (histogram[i] > 0u) { far_idx = i; break; } }
    } else {
        for (var i = 0u; i < 256u; i++) { if (histogram[i] > 0u) { far_idx = i; break; } }
    }
    // Line distance method
    let x1 = f32(peak_idx); let y1 = f32(peak_val);
    let x2 = f32(far_idx); let y2 = f32(histogram[far_idx]);
    let line_len = max(sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)), 0.000001);
    var best_t = peak_idx;
    var max_dist = 0.0;
    let lo = min(peak_idx, far_idx);
    let hi = max(peak_idx, far_idx);
    for (var t = lo; t <= hi; t++) {
        let d = abs((y2-y1)*f32(t) - (x2-x1)*f32(histogram[t]) + x2*y1 - y2*x1) / line_len;
        if (d > max_dist) { max_dist = d; best_t = t; }
    }
    let threshold = f32(best_t) / 255.0;

    let pixel = input[idx];
    let l = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
    let v = select(0.0, 1.0, l >= threshold);
    output[idx] = vec4(v, v, v, pixel.w);
}
"#;
