//! Filter: median (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Median filter with given radius. Window is (2*radius+1)^2.
///
/// Uses histogram sliding-window (Huang algorithm) for radius > 2 giving
/// O(1) amortized per pixel. Falls back to sorting for radius <= 2 where
/// the small window makes sorting faster than histogram maintenance.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "median", category = "spatial",
    group = "denoise",
    variant = "median",
    reference = "median rank filter"
)]
pub struct MedianParams {
    /// Filter radius in pixels
    #[param(min = 1, max = 20, step = 1, default = 3, hint = "rc.log_slider")]
    pub radius: u32,
}

impl CpuFilter for MedianParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let overlap = self.radius;
        let expanded = request.expand_uniform(overlap, info.width, info.height);
        let pixels = upstream(expanded)?;
        let info = &ImageInfo {
            width: expanded.width,
            height: expanded.height,
            ..*info
        };
        let pixels = pixels.as_slice();
        let radius = self.radius;

        if radius == 0 {
            return Ok(pixels.to_vec());
        }
        validate_format(info.format)?;

        if is_16bit(info.format) {
            let cfg = self.clone();
            let result = process_via_8bit(pixels, info, |p8, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                let mut u = |_: Rect| Ok(p8.to_vec());
                cfg.compute(r, &mut u, i8)
            })?;
            return Ok(crop_to_request(&result, expanded, request, info.format));
        }

        let w = info.width as usize;
        let h = info.height as usize;
        let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;

        let result = if radius <= 2 {
            median_sort(pixels, w, h, channels, radius)?
        } else {
            median_histogram(pixels, w, h, channels, radius)?
        };
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.radius;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

impl GpuFilter for MedianParams {
    fn gpu_ops(
        &self,
        width: u32,
        height: u32,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        use rasmcore_gpu_shaders as shaders;

        static MEDIAN: LazyLock<String> =
            LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/median.wgsl")));

        if self.radius > 3 {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.radius.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());

        Some(vec![GpuOp::Compute {
            shader: MEDIAN.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}
