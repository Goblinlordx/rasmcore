//! Filter: gaussian_noise (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Gaussian noise — adds normally-distributed noise to an image
pub struct GaussianNoiseParams {
    /// Noise amount (0 = identity, 100 = full strength)
    #[param(min = 0.0, max = 100.0, step = 0.5, default = 10.0)]
    pub amount: f32,
    /// Mean of the Gaussian distribution (-128 to 128)
    #[param(min = -128.0, max = 128.0, step = 0.5, default = 0.0)]
    pub mean: f32,
    /// Standard deviation (sigma) of the distribution
    #[param(min = 0.0, max = 100.0, step = 0.5, default = 25.0)]
    pub sigma: f32,
    /// Random seed for reproducibility
    #[param(
        min = 0,
        max = 18446744073709551615,
        step = 1,
        default = 42,
        hint = "rc.seed"
    )]
    pub seed: u64,
}

#[rasmcore_macros::register_filter(
    name = "gaussian_noise",
    category = "effect",
    group = "noise",
    variant = "gaussian",
    reference = "additive Gaussian noise"
)]
pub fn gaussian_noise(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &GaussianNoiseParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            gaussian_noise(r, &mut u, i8, config)
        });
    }

    let amount = config.amount as f64 / 100.0;
    if amount == 0.0 {
        return Ok(pixels.to_vec());
    }
    // GPU impl is below — this CPU path continues for fallback

    let mean = config.mean as f64;
    let sigma = config.sigma as f64;
    let ch = channels(info.format);
    let has_alpha = matches!(info.format, PixelFormat::Rgba8);
    let mut rng = config.seed.max(1); // avoid zero state

    let mut out = pixels.to_vec();
    for pixel in out.chunks_exact_mut(ch) {
        let color_ch = if has_alpha { ch - 1 } else { ch };
        for c in &mut pixel[..color_ch] {
            let noise = box_muller(&mut rng) * sigma + mean;
            let v = *c as f64 + noise * amount;
            *c = v.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

impl crate::domain::filter_traits::GpuFilter for GaussianNoiseParams {
    fn gpu_ops(&self, _width: u32, _height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        None
    }

    fn gpu_ops_with_format(
        &self,
        width: u32,
        height: u32,
        buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        if buffer_format != rasmcore_pipeline::gpu::BufferFormat::F32Vec4 {
            return None;
        }
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        static SHADER: LazyLock<String> = LazyLock::new(|| {
            include_str!("../../../shaders/gaussian_noise_f32.wgsl").to_string()
        });
        // Normalize params to [0,1] range matching the f32 pipeline
        let amount_norm = self.amount / 100.0;
        let mean_norm = self.mean / 255.0;
        let sigma_norm = self.sigma / 255.0;
        let seed32 = (self.seed & 0xFFFF_FFFF) as u32;
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&amount_norm.to_le_bytes());
        params.extend_from_slice(&mean_norm.to_le_bytes());
        params.extend_from_slice(&sigma_norm.to_le_bytes());
        params.extend_from_slice(&seed32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        Some(vec![GpuOp::Compute {
            shader: SHADER.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::BufferFormat::F32Vec4,
        }])
    }
}
