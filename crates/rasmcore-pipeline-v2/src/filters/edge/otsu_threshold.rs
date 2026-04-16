//! OtsuThreshold filter.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

use super::super::helpers::luminance;

/// Otsu's method — automatic threshold that maximizes between-class variance.
/// Computes optimal threshold from the image histogram, then applies binary threshold.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "otsu_threshold", category = "adjustment", cost = "O(n)")]
pub struct OtsuThreshold;

impl Filter for OtsuThreshold {
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
        let npx = input.len() / 4;
        // Build 256-bin luminance histogram
        let mut hist = [0u32; 256];
        for px in input.chunks_exact(4) {
            let l = luminance(px[0], px[1], px[2]);
            let bin = ((l * 255.0).round().max(0.0) as usize).min(255);
            hist[bin] += 1;
        }
        // Otsu's algorithm
        let total = npx as f64;
        let mut sum_total = 0.0f64;
        for (i, &h) in hist.iter().enumerate() {
            sum_total += i as f64 * h as f64;
        }
        let mut sum_bg = 0.0f64;
        let mut w_bg = 0.0f64;
        let mut max_var = 0.0f64;
        let mut best_t = 0usize;
        for (t, &h) in hist.iter().enumerate() {
            w_bg += h as f64;
            if w_bg == 0.0 {
                continue;
            }
            let w_fg = total - w_bg;
            if w_fg == 0.0 {
                break;
            }
            sum_bg += t as f64 * h as f64;
            let mean_bg = sum_bg / w_bg;
            let mean_fg = (sum_total - sum_bg) / w_fg;
            let var = w_bg * w_fg * (mean_bg - mean_fg).powi(2);
            if var > max_var {
                max_var = var;
                best_t = t;
            }
        }
        // Place threshold between bins (matching OpenCV: pixel > thresh_u8
        // is equivalent to luma_f32 >= (best_t + 0.5) / 255.0)
        let threshold = (best_t as f32 + 0.5) / 255.0;
        // Apply
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
        let pass3 = GpuShader::new(OTSU_APPLY_WGSL.to_string(), "main", [256, 1, 1], params)
            .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);
        Some(vec![passes.pass1, passes.pass2, pass3])
    }
}

/// Otsu GPU apply shader — reads histogram, computes optimal threshold inline, applies.
const OTSU_APPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, total_pixels: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_pixels) { return; }

    // Compute Otsu threshold from luminance histogram (bins 0..255 of R channel)
    var sum_total = 0.0;
    for (var i = 0u; i < 256u; i++) { sum_total += f32(i) * f32(histogram[i]); }
    var sum_bg = 0.0;
    var w_bg = 0.0;
    var max_var = 0.0;
    var best_t = 0u;
    let total = f32(params.total_pixels);
    for (var t = 0u; t < 256u; t++) {
        w_bg += f32(histogram[t]);
        if (w_bg == 0.0) { continue; }
        let w_fg = total - w_bg;
        if (w_fg == 0.0) { break; }
        sum_bg += f32(t) * f32(histogram[t]);
        let mean_bg = sum_bg / w_bg;
        let mean_fg = (sum_total - sum_bg) / w_fg;
        let v = w_bg * w_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (v > max_var) { max_var = v; best_t = t; }
    }
    let threshold = (f32(best_t) + 0.5) / 255.0;

    let pixel = input[idx];
    let l = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
    let v = select(0.0, 1.0, l >= threshold);
    output[idx] = vec4(v, v, v, pixel.w);
}
"#;
