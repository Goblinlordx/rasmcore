use super::{apply_rgb_transform, ImageError, ImageInfo};

/// ASC Color Decision List parameters — per-channel slope, offset, power.
///
/// Standard reference: ASC CDL specification (Society of Motion Picture and
/// Television Engineers). Formula:
/// ```text
///   out = clamp01((in * slope + offset) ^ power)
/// ```
/// Also supports an overall saturation adjustment.
#[derive(Debug, Clone, Copy)]
pub struct AscCdl {
    pub slope: [f32; 3],
    pub offset: [f32; 3],
    pub power: [f32; 3],
    /// Overall saturation (1.0 = unchanged). Applied after SOP.
    pub saturation: f32,
}

impl Default for AscCdl {
    fn default() -> Self {
        Self {
            slope: [1.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 1.0,
        }
    }
}

/// Apply ASC-CDL transform to a single pixel (r, g, b in [0, 1]).
#[inline]
pub fn asc_cdl_pixel(r: f32, g: f32, b: f32, cdl: &AscCdl) -> (f32, f32, f32) {
    // SOP: slope, offset, power
    let mut out_r = ((r * cdl.slope[0] + cdl.offset[0]).max(0.0)).powf(cdl.power[0]);
    let mut out_g = ((g * cdl.slope[1] + cdl.offset[1]).max(0.0)).powf(cdl.power[1]);
    let mut out_b = ((b * cdl.slope[2] + cdl.offset[2]).max(0.0)).powf(cdl.power[2]);

    // Saturation adjustment (Rec. 709 luma weights)
    if cdl.saturation != 1.0 {
        let luma = 0.2126 * out_r + 0.7152 * out_g + 0.0722 * out_b;
        out_r = luma + (out_r - luma) * cdl.saturation;
        out_g = luma + (out_g - luma) * cdl.saturation;
        out_b = luma + (out_b - luma) * cdl.saturation;
    }

    (
        out_r.clamp(0.0, 1.0),
        out_g.clamp(0.0, 1.0),
        out_b.clamp(0.0, 1.0),
    )
}

/// Apply ASC-CDL to an image pixel buffer.
pub fn asc_cdl(pixels: &[u8], info: &ImageInfo, cdl: &AscCdl) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| asc_cdl_pixel(r, g, b, cdl))
}
