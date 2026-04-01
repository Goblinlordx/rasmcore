//! Filter: apply_cube_lut (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Apply a .cube 3D LUT to the image.
///
/// The `cube_data` parameter is the full text content of the .cube file.
/// The LUT is parsed and applied via tetrahedral interpolation.
#[rasmcore_macros::register_filter(
    name = "apply_cube_lut",
    category = "grading",
    group = "lut_import",
    variant = "cube",
    reference = "Adobe/Resolve .cube 3D LUT format"
)]
pub fn apply_cube_lut(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    cube_data: String,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let lut = crate::domain::color_lut::parse_cube_lut(&cube_data)?;
    lut.apply(pixels, info)
}
