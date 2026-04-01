//! Filter: exposure (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Photoshop-style exposure adjustment — logarithmic brightness with offset and gamma.
///
/// Uses the composable LUT infrastructure from `point_ops`. Fully LUT-collapsible.
#[rasmcore_macros::register_filter(
    name = "exposure",
    category = "adjustment",
    reference = "Photoshop exposure (EV stops + offset + gamma)",
    point_op = "true"
)]
pub fn exposure(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ExposureParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    if config.gamma_correction <= 0.0 {
        return Err(ImageError::InvalidParameters(
            "exposure gamma_correction must be > 0".into(),
        ));
    }
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            exposure(r, &mut u, i8, config)
        });
    }
    crate::domain::point_ops::exposure(
        pixels,
        info,
        config.ev,
        config.offset,
        config.gamma_correction,
    )
}
