//! Filter: charcoal (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Charcoal sketch: Sobel edge detection → blur → invert.
/// IM's -charcoal uses a different edge detector (not Sobel) plus normalize;
/// we use Sobel which produces visually similar but numerically different
/// edge maps. The normalize step is intentionally omitted because it
/// amplifies the edge detector difference (MAE 24→239 with normalize).
/// Registered as mapper because it changes pixel format (RGB8 → Gray8).

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Charcoal sketch effect — edge detection + blur + invert
pub struct CharcoalParams {
    /// Blur radius for smoothing the edge image
    #[param(min = 0.0, max = 10.0, step = 0.1, default = 1.0)]
    pub radius: f32,
    /// Edge detection sensitivity (Gaussian sigma for Sobel pre-blur)
    #[param(min = 0.1, max = 5.0, step = 0.1, default = 0.5)]
    pub sigma: f32,
}

impl rasmcore_pipeline::GpuCapable for CharcoalParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        use std::sync::LazyLock;
        static CHARCOAL_F32: LazyLock<String> = LazyLock::new(|| {
            rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!(
                "../../../shaders/charcoal_f32.wgsl"
            ))
        });

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.radius.to_le_bytes());
        params.extend_from_slice(&self.sigma.to_le_bytes());

        // GPU shader outputs grayscale as RGBA f32 (same luma in R/G/B).
        // The pipeline handles format conversion to Gray8 at the boundary.
        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: CHARCOAL_F32.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: rasmcore_pipeline::gpu::BufferFormat::F32Vec4,
        }])
    }
}

#[rasmcore_macros::register_mapper(
    name = "charcoal",
    category = "effect",
    reference = "charcoal drawing edge effect",
    output_format = "Gray8"
)]
pub fn charcoal(
    pixels: &[u8],
    info: &ImageInfo,
    config: &CharcoalParams,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let radius = config.radius;
    let sigma = config.sigma;

    // 1. Optional pre-blur to control edge sensitivity
    let smoothed = if sigma > 0.0 {
        blur_impl(pixels, info, &BlurParams { radius: sigma })?
    } else {
        pixels.to_vec()
    };

    // 2. Edge detection via Sobel — outputs Gray8
    let edges = sobel(&smoothed, info)?;
    let gray_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };

    // 3. Post-blur to soften the edges (on the grayscale edge image)
    let blurred = if radius > 0.0 {
        blur_impl(&edges, &gray_info, &BlurParams { radius })?
    } else {
        edges
    };

    // 4. Invert to get dark lines on white background
    let result = crate::domain::point_ops::invert(&blurred, &gray_info)?;
    Ok((result, gray_info))
}
