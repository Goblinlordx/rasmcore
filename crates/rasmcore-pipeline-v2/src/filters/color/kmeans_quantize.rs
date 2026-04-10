use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::{kmeans_palette, nearest_color};

/// K-means color quantization.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(
    name = "kmeans_quantize",
    category = "color",
    cost = "O(n * k * iterations)"
)]
pub struct KmeansQuantize {
    /// Number of clusters (2-256).
    #[param(min = 2, max = 256, default = 16)]
    pub k: u32,
    /// Maximum iterations.
    #[param(min = 1, max = 100, default = 20)]
    pub max_iterations: u32,
    /// Random seed for deterministic initialization.
    #[param(min = 0, max = 100, default = 42)]
    pub seed: u32,
}

impl Filter for KmeansQuantize {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let palette = kmeans_palette(input, self.k as usize, self.max_iterations, self.seed);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = nearest_color(&palette, pixel[0], pixel[1], pixel[2]);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }
}

impl GpuFilter for KmeansQuantize {
    fn shader_body(&self) -> &str {
        include_str!("../../shaders/kmeans_quantize.wgsl")
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [256, 1, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.k.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Palette is computed by the CPU compute() path then passed here.
        // For GPU dispatch, the pipeline calls compute() first to get the palette,
        // then re-encodes it. This is handled by the GpuFilterNode wrapper.
        // Empty here — palette injected by the caller.
        vec![]
    }
}
