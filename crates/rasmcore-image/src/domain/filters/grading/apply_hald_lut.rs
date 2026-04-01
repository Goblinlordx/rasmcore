//! Filter: apply_hald_lut (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ApplyHaldLutParams {
    pub hald_dim: u32,
}

#[rasmcore_macros::register_filter(
    name = "apply_hald_lut",
    category = "grading",
    group = "lut_import",
    variant = "hald",
    reference = "ImageMagick HALD CLUT format"
)]
pub fn apply_hald_lut(
    request: Rect,
    upstream: &mut UpstreamFn,
    _info: &ImageInfo,
    _config: &ApplyHaldLutParams,
) -> Result<Vec<u8>, ImageError> {
    let _pixels = upstream(request)?;
    let _info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*_info
    };
    let _pixels = _pixels.as_slice();

    // For the registered filter, hald_dim is a placeholder — the actual HALD
    // pixel data must be provided programmatically via parse_hald_lut + apply.
    // This filter exists for manifest/registry purposes; the pipeline dispatches
    // HALD application through the domain API directly.
    Err(ImageError::InvalidParameters(
        "apply_hald_lut requires HALD image data — use the pipeline API with parse_hald_lut() + ColorLut3D::apply()".into(),
    ))
}
