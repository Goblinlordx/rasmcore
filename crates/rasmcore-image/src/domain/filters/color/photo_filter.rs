//! Filter: photo_filter (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Apply a photo filter (warming/cooling color overlay).
///
/// Blends a solid color over the image at the given density. When
/// preserve_luminosity is enabled, the original pixel's luminance is
/// maintained (only hue/saturation shifts). PS Photo Filter equivalent.
#[derive(rasmcore_macros::Filter, Clone)]
/// Photo Filter — warming/cooling color overlay like a camera lens filter.
#[filter(name = "photo_filter", category = "color")]
pub struct PhotoFilterParams {
    /// Filter color red
    #[param(min = 0, max = 255, step = 1, default = 236)]
    pub color_r: u32,
    /// Filter color green
    #[param(min = 0, max = 255, step = 1, default = 138)]
    pub color_g: u32,
    /// Filter color blue
    #[param(min = 0, max = 255, step = 1, default = 0)]
    pub color_b: u32,
    /// Filter density (0 = no effect, 100 = full color replacement)
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 25.0)]
    pub density: f32,
    /// Preserve luminosity (keep original brightness)
    #[param(min = 0, max = 1, step = 1, default = 1)]
    pub preserve_luminosity: u32,
}

impl CpuFilter for PhotoFilterParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    let color_r = self.color_r;
    let color_g = self.color_g;
    let color_b = self.color_b;
    let density = self.density;
    let preserve_luminosity = self.preserve_luminosity;

    let density = (density / 100.0).clamp(0.0, 1.0);
    if density == 0.0 {
        return Ok(pixels.to_vec());
    }

    let fr = color_r.min(255) as f32 / 255.0;
    let fg = color_g.min(255) as f32 / 255.0;
    let fb = color_b.min(255) as f32 / 255.0;
    let preserve = preserve_luminosity != 0;

    crate::domain::color_grading::apply_rgb_transform(pixels, info, |r, g, b| {
        let mut nr = r + (fr - r) * density;
        let mut ng = g + (fg - g) * density;
        let mut nb = b + (fb - b) * density;

        if preserve {
            let orig_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let new_luma = 0.2126 * nr + 0.7152 * ng + 0.0722 * nb;
            if new_luma > 0.0 {
                let scale = orig_luma / new_luma;
                nr = (nr * scale).clamp(0.0, 1.0);
                ng = (ng * scale).clamp(0.0, 1.0);
                nb = (nb * scale).clamp(0.0, 1.0);
            }
        }

        (nr, ng, nb)
    })
}
}

impl GpuFilter for PhotoFilterParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::gpu::BufferFormat::U32Packed)
    }

    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        use rasmcore_pipeline::gpu::{BufferFormat, GpuOp};
        use std::sync::LazyLock;

        static SHADER_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/photo_filter_f32.wgsl"
            ))
        });

        if buffer_format != BufferFormat::F32Vec4 { return None; }

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&(self.color_r.min(255) as f32 / 255.0).to_le_bytes());
        params.extend_from_slice(&(self.color_g.min(255) as f32 / 255.0).to_le_bytes());
        params.extend_from_slice(&(self.color_b.min(255) as f32 / 255.0).to_le_bytes());
        params.extend_from_slice(&(self.density / 100.0).to_le_bytes());
        params.extend_from_slice(&self.preserve_luminosity.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // pad

        Some(vec![GpuOp::Compute {
            shader: SHADER_F32.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: BufferFormat::F32Vec4,
        }])
    }
}

