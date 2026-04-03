//! V2 encoders — accept f32 RGBA, output encoded bytes.
//!
//! Each encoder handles view transform (Linear→sRGB) and quantization (f32→u8)
//! internally for sRGB formats. Linear formats (EXR, HDR, FITS) write f32
//! directly without gamma encoding.

use crate::convert;
use rasmcore_pipeline_v2::ops::Encoder;
use rasmcore_pipeline_v2::registry::{
    OperationCapabilities, OperationKind, OperationRegistration, ParamDescriptor, ParamType,
};
use rasmcore_pipeline_v2::PipelineError;

// ─── Macro for sRGB encoders (f32 → gamma → quantize → encode) ──────────────

macro_rules! v2_srgb_encoder {
    (
        struct_name: $struct:ident,
        format: $format:literal,
        display_name: $display:literal,
        mime: $mime:literal,
        extensions: [$($ext:literal),+],
        alpha: $alpha:expr,
        reg_name: $reg_name:ident
    ) => {
        pub struct $struct;

        impl Encoder for $struct {
            fn encode(
                &self,
                pixels: &[f32],
                width: u32,
                height: u32,
                quality: Option<u8>,
            ) -> Result<Vec<u8>, PipelineError> {
                if $alpha {
                    // Format supports alpha — use Rgba8
                    let (u8_pixels, info) = convert::f32_to_v1_rgba8(pixels, width, height, true);
                    rasmcore_image::domain::encoder::encode(&u8_pixels, &info, $format, quality)
                        .map_err(|e| PipelineError::ComputeError(format!("{}: {e}", $format)))
                } else {
                    // Format doesn't support alpha — use Rgb8
                    let (u8_pixels, info) = convert::f32_to_v1_rgb8(pixels, width, height, true);
                    rasmcore_image::domain::encoder::encode(&u8_pixels, &info, $format, quality)
                        .map_err(|e| PipelineError::ComputeError(format!("{}: {e}", $format)))
                }
            }

            fn mime_type(&self) -> &str {
                $mime
            }

            fn extensions(&self) -> &[&str] {
                &[$($ext),+]
            }
        }

        inventory::submit! {
            &OperationRegistration {
                name: concat!($format, "_encode"),
                display_name: concat!($display, " Encoder"),
                category: "codec",
                kind: OperationKind::Encoder,
                params: &QUALITY_PARAM,
                capabilities: OperationCapabilities {
                    gpu: false, analytic: false, affine: false, clut: false,
                },
            }
        }
    };
}

// ─── Macro for linear f32 encoders (EXR, HDR, FITS) ─────────────────────────

macro_rules! v2_linear_encoder {
    (
        struct_name: $struct:ident,
        format: $format:literal,
        display_name: $display:literal,
        mime: $mime:literal,
        extensions: [$($ext:literal),+],
        reg_name: $reg_name:ident
    ) => {
        pub struct $struct;

        impl Encoder for $struct {
            fn encode(
                &self,
                pixels: &[f32],
                width: u32,
                height: u32,
                quality: Option<u8>,
            ) -> Result<Vec<u8>, PipelineError> {
                // Write f32 directly — no gamma encoding
                let (f32_bytes, info) = convert::f32_to_v1_rgba32f_bytes(pixels, width, height);
                rasmcore_image::domain::encoder::encode(&f32_bytes, &info, $format, quality)
                    .map_err(|e| PipelineError::ComputeError(format!("{}: {e}", $format)))
            }

            fn mime_type(&self) -> &str {
                $mime
            }

            fn extensions(&self) -> &[&str] {
                &[$($ext),+]
            }
        }

        inventory::submit! {
            &OperationRegistration {
                name: concat!($format, "_encode"),
                display_name: concat!($display, " Encoder"),
                category: "codec",
                kind: OperationKind::Encoder,
                params: &[],
                capabilities: OperationCapabilities {
                    gpu: false, analytic: false, affine: false, clut: false,
                },
            }
        }
    };
}

// ─── Shared param descriptor ─────────────────────────────────────────────────

static QUALITY_PARAM: [ParamDescriptor; 1] = [ParamDescriptor {
    name: "quality",
    value_type: ParamType::U32,
    min: Some(1.0),
    max: Some(100.0),
    step: Some(1.0),
    default: Some(85.0),
    hint: Some("slider"),
    constraints: &[],
}];

// ═══════════════════════════════════════════════════════════════════════════════
// sRGB encoders — f32 Linear → sRGB gamma → u8 → encode
// ═══════════════════════════════════════════════════════════════════════════════

v2_srgb_encoder! {
    struct_name: PngEncoder,
    format: "png",
    display_name: "PNG",
    mime: "image/png",
    extensions: ["png"],
    alpha: true,
    reg_name: PNG_ENCODE_REG
}

v2_srgb_encoder! {
    struct_name: JpegEncoder,
    format: "jpeg",
    display_name: "JPEG",
    mime: "image/jpeg",
    extensions: ["jpg", "jpeg", "jfif"],
    alpha: false,
    reg_name: JPEG_ENCODE_REG
}

v2_srgb_encoder! {
    struct_name: WebpEncoder,
    format: "webp",
    display_name: "WebP",
    mime: "image/webp",
    extensions: ["webp"],
    alpha: true,
    reg_name: WEBP_ENCODE_REG
}

v2_srgb_encoder! {
    struct_name: GifEncoder,
    format: "gif",
    display_name: "GIF",
    mime: "image/gif",
    extensions: ["gif"],
    alpha: true,
    reg_name: GIF_ENCODE_REG
}

v2_srgb_encoder! {
    struct_name: BmpEncoder,
    format: "bmp",
    display_name: "BMP",
    mime: "image/bmp",
    extensions: ["bmp", "dib"],
    alpha: false,
    reg_name: BMP_ENCODE_REG
}

v2_srgb_encoder! {
    struct_name: QoiEncoder,
    format: "qoi",
    display_name: "QOI",
    mime: "image/x-qoi",
    extensions: ["qoi"],
    alpha: true,
    reg_name: QOI_ENCODE_REG
}

v2_srgb_encoder! {
    struct_name: IcoEncoder,
    format: "ico",
    display_name: "ICO",
    mime: "image/x-icon",
    extensions: ["ico", "cur"],
    alpha: true,
    reg_name: ICO_ENCODE_REG
}

v2_srgb_encoder! {
    struct_name: TgaEncoder,
    format: "tga",
    display_name: "TGA",
    mime: "image/x-tga",
    extensions: ["tga", "targa"],
    alpha: true,
    reg_name: TGA_ENCODE_REG
}

v2_srgb_encoder! {
    struct_name: PnmEncoder,
    format: "pnm",
    display_name: "PNM",
    mime: "image/x-portable-anymap",
    extensions: ["pnm", "ppm"],
    alpha: false,
    reg_name: PNM_ENCODE_REG
}

// TIFF supports both u8 sRGB and f32 output — default to sRGB u8 Rgb8 for
// maximum compatibility. TIFF f32 output can be added via a separate
// TiffF32Encoder if needed.
v2_srgb_encoder! {
    struct_name: TiffEncoder,
    format: "tiff",
    display_name: "TIFF",
    mime: "image/tiff",
    extensions: ["tiff", "tif"],
    alpha: false,
    reg_name: TIFF_ENCODE_REG
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linear f32 encoders — write f32 directly (no gamma encoding)
// ═══════════════════════════════════════════════════════════════════════════════

v2_linear_encoder! {
    struct_name: ExrEncoder,
    format: "exr",
    display_name: "OpenEXR",
    mime: "image/x-exr",
    extensions: ["exr"],
    reg_name: EXR_ENCODE_REG
}

v2_linear_encoder! {
    struct_name: HdrEncoder,
    format: "hdr",
    display_name: "Radiance HDR",
    mime: "image/vnd.radiance",
    extensions: ["hdr", "rgbe"],
    reg_name: HDR_ENCODE_REG
}

// FITS requires special handling — grayscale f32, not the generic linear encoder.
pub struct FitsEncoder;

impl Encoder for FitsEncoder {
    fn encode(
        &self,
        pixels: &[f32],
        width: u32,
        height: u32,
        _quality: Option<u8>,
    ) -> Result<Vec<u8>, PipelineError> {
        // FITS is typically grayscale — extract luma from RGBA f32
        let npixels = (width as usize) * (height as usize);
        let mut gray = Vec::with_capacity(npixels);
        for chunk in pixels.chunks_exact(4) {
            // Rec. 709 luma coefficients
            let luma = 0.2126 * chunk[0] + 0.7152 * chunk[1] + 0.0722 * chunk[2];
            gray.push(luma);
        }
        rasmcore_fits::encode_f32(&gray, width, height)
            .map_err(|e| PipelineError::ComputeError(format!("fits: {e}")))
    }

    fn mime_type(&self) -> &str {
        "image/fits"
    }

    fn extensions(&self) -> &[&str] {
        &["fits", "fit"]
    }
}

inventory::submit! {
    &OperationRegistration {
        name: "fits_encode",
        display_name: "FITS Encoder",
        category: "codec",
        kind: OperationKind::Encoder,
        params: &[],
        capabilities: OperationCapabilities {
            gpu: false, analytic: false, affine: false, clut: false,
        },
    }
}

// ─── V2 encoder dispatch ────────────────────────────────────────────────────

/// Find a V2 encoder by format name and encode f32 pixels.
pub fn encode(
    pixels: &[f32],
    width: u32,
    height: u32,
    format: &str,
    quality: Option<u8>,
) -> Result<Vec<u8>, PipelineError> {
    let encoder: &dyn Encoder = match format {
        "png" => &PngEncoder,
        "jpeg" | "jpg" | "jfif" => &JpegEncoder,
        "webp" => &WebpEncoder,
        "gif" => &GifEncoder,
        "bmp" | "dib" => &BmpEncoder,
        "qoi" => &QoiEncoder,
        "ico" | "cur" => &IcoEncoder,
        "tga" | "targa" => &TgaEncoder,
        "tiff" | "tif" => &TiffEncoder,
        "pnm" | "ppm" => &PnmEncoder,
        "exr" => &ExrEncoder,
        "hdr" | "rgbe" => &HdrEncoder,
        "fits" | "fit" => &FitsEncoder,
        _ => {
            return Err(PipelineError::ComputeError(format!(
                "no V2 encoder for format '{format}'"
            )));
        }
    };
    encoder.encode(pixels, width, height, quality)
}

/// List all supported V2 encode formats.
pub fn supported_formats() -> Vec<&'static str> {
    vec![
        "png", "jpeg", "webp", "gif", "bmp", "qoi", "ico", "tga",
        "tiff", "pnm", "exr", "hdr", "fits",
    ]
}

/// Check if a format is supported for V2 encoding.
pub fn is_supported(format: &str) -> bool {
    matches!(
        format,
        "png" | "jpeg" | "jpg" | "jfif" | "webp" | "gif" | "bmp" | "dib"
            | "qoi" | "ico" | "cur" | "tga" | "targa" | "tiff" | "tif"
            | "pnm" | "ppm" | "exr" | "hdr" | "rgbe" | "fits" | "fit"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_encode_formats() {
        let fmts = supported_formats();
        assert_eq!(fmts.len(), 13);
        assert!(fmts.contains(&"png"));
        assert!(fmts.contains(&"exr"));
    }

    #[test]
    fn unknown_format_returns_error() {
        let pixels = vec![0.5f32; 4]; // 1x1 RGBA
        assert!(encode(&pixels, 1, 1, "foobar", None).is_err());
    }

    #[test]
    fn is_supported_checks() {
        assert!(is_supported("png"));
        assert!(is_supported("jpeg"));
        assert!(is_supported("jpg"));
        assert!(is_supported("exr"));
        assert!(!is_supported("foobar"));
    }
}
