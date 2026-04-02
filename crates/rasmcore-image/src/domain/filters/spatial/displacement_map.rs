//! Filter: displacement_map (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Warp an image by a per-pixel displacement field, matching OpenCV `cv2.remap`
/// with `INTER_LINEAR` interpolation and `BORDER_CONSTANT` (value=0).
///
/// `map_x` and `map_y` are f32 slices of length `width * height`. For each
/// output pixel `(x, y)`, the source is sampled at `(map_x[y*w+x], map_y[y*w+x])`
/// using bilinear interpolation. Out-of-bounds source coordinates produce black
/// (zero) pixels.
///
/// Supports RGB8, RGBA8, Gray8. 16-bit formats are processed via 8-bit downscale.
#[rasmcore_macros::register_filter(
    name = "displacement_map",
    category = "spatial",
    reference = "displacement mapping"
)]
pub fn displacement_map(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    map_x: &[f32],
    map_y: &[f32],
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let n = w * h;

    if map_x.len() != n || map_y.len() != n {
        return Err(ImageError::InvalidParameters(format!(
            "displacement map size mismatch: expected {}x{}={}, got map_x={} map_y={}",
            w,
            h,
            n,
            map_x.len(),
            map_y.len()
        )));
    }

    // Convert to f32 samples [0,1] for format-agnostic processing
    let samples = pixels_to_f32_samples(pixels, info.format);
    let mut out = vec![0.0f32; samples.len()];
    let wi = w as i32;
    let hi = h as i32;

    for y in 0..h {
        let row_off = y * w;
        for x in 0..w {
            let idx = row_off + x;
            let sx = map_x[idx];
            let sy = map_y[idx];

            // Entirely outside the bilinear footprint -> 0 (already zeroed)
            if sx < -1.0 || sy < -1.0 || sx >= wi as f32 || sy >= hi as f32 {
                continue;
            }

            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            // Inline helper: fetch sample or 0 if out-of-bounds (BORDER_CONSTANT)
            let sample_val = |px: i32, py: i32, c: usize| -> f32 {
                if px >= 0 && px < wi && py >= 0 && py < hi {
                    samples[(py as usize * w + px as usize) * ch + c]
                } else {
                    0.0
                }
            };

            let out_off = idx * ch;
            for c in 0..ch {
                let v = sample_val(x0, y0, c) * w00
                    + sample_val(x1, y0, c) * w10
                    + sample_val(x0, y1, c) * w01
                    + sample_val(x1, y1, c) * w11;
                out[out_off + c] = v;
            }
        }
    }

    Ok(f32_samples_to_pixels(&out, info.format))
}
