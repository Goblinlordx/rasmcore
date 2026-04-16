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
        let mut hist = [0i32; 256];
        for px in input.chunks_exact(4) {
            let l = luminance(px[0], px[1], px[2]);
            let bin = ((l * 255.0).round().max(0.0) as usize).min(255);
            hist[bin] += 1;
        }

        // Matching OpenCV's getThreshVal_Triangle_8u exactly:

        // Find left/right bounds (extend by 1 bin)
        let mut left_bound = 0i32;
        for i in 0..256 {
            if hist[i] > 0 { left_bound = i as i32; break; }
        }
        if left_bound > 0 { left_bound -= 1; }

        let mut right_bound = 0i32;
        for i in (0..256).rev() {
            if hist[i] > 0 { right_bound = i as i32; break; }
        }
        if right_bound < 255 { right_bound += 1; }

        // Find peak (strict >, first max wins)
        let mut max_ind = 0i32;
        let mut max_val = 0i32;
        for i in 0..256 {
            if hist[i] > max_val {
                max_val = hist[i];
                max_ind = i as i32;
            }
        }

        // Flip if peak is closer to left
        let mut isflipped = false;
        if max_ind - left_bound < right_bound - max_ind {
            isflipped = true;
            hist.reverse();
            left_bound = 255 - right_bound;
            max_ind = 255 - max_ind;
        }

        // Simplified distance: a*i + b*h[i] (no normalization needed)
        let a = max_val as f64;
        let b = (left_bound - max_ind) as f64;
        let mut thresh = left_bound;
        let mut dist = 0.0f64;
        for i in (left_bound + 1)..=max_ind {
            let tempdist = a * i as f64 + b * hist[i as usize] as f64;
            if tempdist > dist {
                dist = tempdist;
                thresh = i;
            }
        }
        thresh -= 1; // OpenCV decrements by 1

        if isflipped {
            thresh = 255 - thresh;
        }

        // Apply threshold with +0.5 between-bin placement
        let threshold = (thresh as f32 + 0.5) / 255.0;
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
    let threshold = (f32(best_t) + 0.5) / 255.0;

    let pixel = input[idx];
    let l = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
    let v = select(0.0, 1.0, l >= threshold);
    output[idx] = vec4(v, v, v, pixel.w);
}
"#;
