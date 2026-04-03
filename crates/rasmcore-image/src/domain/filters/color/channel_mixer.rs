//! Filter: channel_mixer (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Mix RGB channels via a 3x3 matrix for creative color grading.
///
/// Identity matrix (1,0,0, 0,1,0, 0,0,1) produces unchanged output.
/// IM equivalent: `-color-matrix "rr rg rb 0 / gr gg gb 0 / br bg bb 0 / 0 0 0 1"`

#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Channel mixer — cross-mix RGB channels via a 3x3 matrix.
pub struct ChannelMixerParams {
    /// Red-from-Red weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 1.0, hint = "rc.signed_slider")]
    pub rr: f32,
    /// Red-from-Green weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub rg: f32,
    /// Red-from-Blue weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub rb: f32,
    /// Green-from-Red weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub gr: f32,
    /// Green-from-Green weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 1.0, hint = "rc.signed_slider")]
    pub gg: f32,
    /// Green-from-Blue weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub gb: f32,
    /// Blue-from-Red weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub br: f32,
    /// Blue-from-Green weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub bg: f32,
    /// Blue-from-Blue weight
    #[param(min = -2.0, max = 2.0, step = 0.01, default = 1.0, hint = "rc.signed_slider")]
    pub bb: f32,
}
impl ColorLutOp for ChannelMixerParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::ChannelMix([
            self.rr, self.rg, self.rb, self.gr, self.gg, self.gb, self.br, self.bg, self.bb,
        ])
        .to_clut(DEFAULT_CLUT_GRID)
    }
}

#[rasmcore_macros::register_filter(
    name = "channel_mixer",
    category = "color",
    reference = "RGB channel matrix multiplication",
    color_op = "true"
)]
#[allow(clippy::too_many_arguments)]
pub fn channel_mixer(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ChannelMixerParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let rr = config.rr;
    let rg = config.rg;
    let rb = config.rb;
    let gr = config.gr;
    let gg = config.gg;
    let gb = config.gb;
    let br = config.br;
    let bg = config.bg;
    let bb = config.bb;

    apply_color_op(
        pixels,
        info,
        &ColorOp::ChannelMix([rr, rg, rb, gr, gg, gb, br, bg, bb]),
    )
}

impl crate::domain::filter_traits::GpuFilter for ChannelMixerParams {
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
            include_str!("../../../shaders/channel_mixer_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(48);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.rr.to_le_bytes());
        params.extend_from_slice(&self.rg.to_le_bytes());
        params.extend_from_slice(&self.rb.to_le_bytes());
        params.extend_from_slice(&self.gr.to_le_bytes());
        params.extend_from_slice(&self.gg.to_le_bytes());
        params.extend_from_slice(&self.gb.to_le_bytes());
        params.extend_from_slice(&self.br.to_le_bytes());
        params.extend_from_slice(&self.bg.to_le_bytes());
        params.extend_from_slice(&self.bb.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // pad to 48 bytes (12 * 4)
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
