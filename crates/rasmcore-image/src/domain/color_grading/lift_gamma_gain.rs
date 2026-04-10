use super::{ImageError, ImageInfo, apply_rgb_transform};

/// 3-way color corrector parameters — lift, gamma, gain per channel.
///
/// DaVinci Resolve formula:
/// ```text
///   out = gain * (input + lift * (1 - input)) ^ (1/gamma)
/// ```
///
/// Neutral values: lift=[0,0,0], gamma=[1,1,1], gain=[1,1,1].
#[derive(Debug, Clone, Copy)]
pub struct LiftGammaGain {
    pub lift: [f32; 3],
    pub gamma: [f32; 3],
    pub gain: [f32; 3],
}

impl Default for LiftGammaGain {
    fn default() -> Self {
        Self {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
        }
    }
}

/// Apply lift/gamma/gain to a single pixel.
#[inline]
pub fn lift_gamma_gain_pixel(r: f32, g: f32, b: f32, lgg: &LiftGammaGain) -> (f32, f32, f32) {
    #[inline]
    fn channel(val: f32, lift: f32, gamma: f32, gain: f32) -> f32 {
        let lifted = val + lift * (1.0 - val);
        let gammaed = if gamma > 0.0 && lifted > 0.0 {
            lifted.powf(1.0 / gamma)
        } else {
            0.0
        };
        (gain * gammaed).clamp(0.0, 1.0)
    }

    (
        channel(r, lgg.lift[0], lgg.gamma[0], lgg.gain[0]),
        channel(g, lgg.lift[1], lgg.gamma[1], lgg.gain[1]),
        channel(b, lgg.lift[2], lgg.gamma[2], lgg.gain[2]),
    )
}

/// Apply lift/gamma/gain to an image pixel buffer.
pub fn lift_gamma_gain(
    pixels: &[u8],
    info: &ImageInfo,
    lgg: &LiftGammaGain,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| lift_gamma_gain_pixel(r, g, b, lgg))
}
