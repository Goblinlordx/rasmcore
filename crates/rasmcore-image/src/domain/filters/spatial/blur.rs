//! Filter: blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Gaussian blur using our own convolve() function.
///
/// Uses separable gaussian convolution (auto-detected by convolve) with
/// WASM SIMD128 acceleration. Large sigma (>= 20) uses box blur
/// approximation for O(1) per-pixel performance.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "blur", category = "spatial",
    group = "blur",
    reference = "Gaussian convolution"
)]
pub struct BlurParams {
    /// Blur radius in pixels
    #[param(
        min = 0.0,
        max = 100.0,
        step = 0.5,
        default = 3.0,
        hint = "rc.log_slider"
    )]
    pub radius: f32,
}

impl CpuFilter for BlurParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let radius = self.radius;
        let overlap = if radius > 0.0 { ((radius * 3.0).ceil() as u32).max(1) } else { 0 };
        let expanded = request.expand_uniform(overlap, info.width, info.height);
        let pixels = upstream(expanded)?;
        let expanded_info = ImageInfo { width: expanded.width, height: expanded.height, ..*info };
        let result = blur_impl(&pixels, &expanded_info, self)?;
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = if self.radius > 0.0 { ((self.radius * 3.0).ceil() as u32).max(1) } else { 0 };
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

impl GpuFilter for BlurParams {
    fn gpu_ops(
        &self,
        width: u32,
        height: u32,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static GAUSSIAN_BLUR: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/gaussian_blur.wgsl")));

        if self.radius <= 0.0 {
            return None;
        }
        let sigma = self.radius;
        let kernel_radius = (sigma * 3.0).ceil() as u32;
        if kernel_radius > 32 {
            return None;
        }

        let ksize = 2 * kernel_radius + 1;
        let mut weights = Vec::with_capacity(ksize as usize);
        let mut sum = 0.0f32;
        for i in 0..ksize {
            let x = i as f32 - kernel_radius as f32;
            let w = (-0.5 * (x / sigma) * (x / sigma)).exp();
            weights.push(w);
            sum += w;
        }
        let inv_sum = 1.0 / sum;
        let mut kernel_buf = Vec::with_capacity(ksize as usize * 4);
        for w in &weights {
            kernel_buf.extend_from_slice(&(w * inv_sum).to_le_bytes());
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&kernel_radius.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        let shader = GAUSSIAN_BLUR.clone();

        Some(vec![
            GpuOp::Compute {
                shader: shader.clone(),
                entry_point: "blur_h",
                workgroup_size: [256, 1, 1],
                params: params.clone(),
                extra_buffers: vec![kernel_buf.clone()],
                buffer_format: Default::default(),
            },
            GpuOp::Compute {
                shader,
                entry_point: "blur_v",
                workgroup_size: [1, 256, 1],
                params,
                extra_buffers: vec![kernel_buf],
                buffer_format: Default::default(),
            },
        ])
    }
}
