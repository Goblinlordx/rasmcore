//! Filter: motion_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "motion_blur", category = "spatial",
    group = "blur",
    variant = "motion",
    reference = "linear kernel simulating camera motion"
)]
pub struct MotionBlurParams {
    #[param(min = 0, max = 100, step = 1, default = 10)]
    pub length: u32,
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 0.0)]
    pub angle_degrees: f32,
}

impl CpuFilter for MotionBlurParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let overlap = self.length;
        let expanded = request.expand_uniform(overlap, info.width, info.height);
        let pixels = upstream(expanded)?;
        let info = &ImageInfo {
            width: expanded.width,
            height: expanded.height,
            ..*info
        };
        let pixels = pixels.as_slice();
        let length = self.length;
        let angle_degrees = self.angle_degrees;

        if length == 0 {
            return Ok(pixels.to_vec());
        }
        validate_format(info.format)?;

        let side = (2 * length + 1) as usize;
        let center = length as f32;
        let angle = angle_degrees.to_radians();
        let dx = angle.cos();
        let dy = -angle.sin();

        let mut kernel = vec![0.0f32; side * side];
        let steps = (length as f32 * 2.0).ceil() as usize * 2 + 1;
        let mut count = 0u32;
        for i in 0..steps {
            let t = (i as f32 / (steps - 1) as f32) * 2.0 - 1.0;
            let px = center + t * length as f32 * dx;
            let py = center + t * length as f32 * dy;
            let ix = px.round() as usize;
            let iy = py.round() as usize;
            if ix < side && iy < side {
                let idx = iy * side + ix;
                if kernel[idx] == 0.0 {
                    kernel[idx] = 1.0;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok(pixels.to_vec());
        }

        let full_rect = Rect::new(0, 0, info.width, info.height);
        let result = {
            let mut u = |_: Rect| Ok(pixels.to_vec());
            convolve(
                full_rect,
                &mut u,
                info,
                &kernel,
                &ConvolveParams {
                    kw: side as u32,
                    kh: side as u32,
                    divisor: count as f32,
                },
            )
        }?;
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.length;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

impl GpuFilter for MotionBlurParams {
    fn gpu_ops(
        &self,
        width: u32,
        height: u32,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static MOTION_BLUR: LazyLock<String> =
            LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/motion_blur.wgsl")));

        if self.length == 0 {
            return None;
        }
        let angle_rad = self.angle_degrees * std::f32::consts::PI / 180.0;

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.length.to_le_bytes());
        params.extend_from_slice(&angle_rad.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: MOTION_BLUR.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}
