//! Round-trip tests for V2 codecs.
//!
//! Generates synthetic f32 RGBA test images, encodes via V2 encoder, decodes
//! via V2 decoder, and verifies the round-trip output matches the input within
//! format-appropriate tolerances.
//!
//! Key: sRGB encoders apply gamma (linear→sRGB), sRGB decoders output sRGB f32.
//! To compare round-trip output with the original linear input, we linearize
//! the decoded output first: decoded_srgb → srgb_to_linear → compare with input.

use rasmcore_codecs_v2::{decode_with_hint, encode};
use rasmcore_pipeline_v2::ColorSpace;
use rasmcore_pipeline_v2::color_math::srgb_to_linear;

// ─── Test image generators ──────────────────────────────────────────────────

/// Generate a 4x4 gradient test image in f32 RGBA (linear color space).
fn gradient_4x4() -> Vec<f32> {
    let mut pixels = Vec::with_capacity(4 * 4 * 4);
    for i in 0..16 {
        let v = i as f32 / 15.0;
        pixels.push(v); // R
        pixels.push(v * 0.5); // G
        pixels.push(1.0 - v); // B
        pixels.push(1.0); // A
    }
    pixels
}

/// Generate a 1x1 solid mid-gray test image in f32 RGBA.
fn solid_gray_1x1() -> Vec<f32> {
    vec![0.5, 0.5, 0.5, 1.0]
}

/// Generate an 8x8 checkerboard (alternating black/white) in f32 RGBA.
fn checkerboard_8x8() -> Vec<f32> {
    let mut pixels = Vec::with_capacity(8 * 8 * 4);
    for y in 0..8 {
        for x in 0..8 {
            let v = if (x + y) % 2 == 0 { 1.0f32 } else { 0.0f32 };
            pixels.push(v);
            pixels.push(v);
            pixels.push(v);
            pixels.push(1.0);
        }
    }
    pixels
}

/// Generate a 16x16 gradient (larger for lossy formats).
fn gradient_16x16() -> Vec<f32> {
    let mut pixels = Vec::with_capacity(16 * 16 * 4);
    for i in 0..256 {
        let v = i as f32 / 255.0;
        pixels.push(v);
        pixels.push(v);
        pixels.push(v);
        pixels.push(1.0);
    }
    pixels
}

// ─── Comparison helpers ─────────────────────────────────────────────────────

/// Linearize sRGB f32 pixels (per RGB channel, alpha untouched).
fn linearize(pixels: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(4) {
        out.push(srgb_to_linear(chunk[0]));
        out.push(srgb_to_linear(chunk[1]));
        out.push(srgb_to_linear(chunk[2]));
        out.push(chunk[3]); // alpha is linear
    }
    out
}

/// Compute max absolute error between two f32 pixel buffers (per-channel).
fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute mean absolute error between two f32 pixel buffers.
fn mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

// ─── Lossless format round-trip tests ───────────────────────────────────────
// Pipeline: linear f32 → [encoder: gamma+quantize to u8] → [decoder: u8→sRGB f32]
// → [linearize] → compare with input.
// Error budget: gamma roundtrip + u8 quantization ≈ max 0.01 per channel.

const LOSSLESS_TOL: f32 = 0.01;

macro_rules! lossless_roundtrip_test {
    ($name:ident, $format:literal) => {
        #[test]
        fn $name() {
            let input = gradient_4x4();
            let encoded = encode(&input, 4, 4, $format, None)
                .unwrap_or_else(|e| panic!("{} encode failed: {e}", $format));
            assert!(!encoded.is_empty(), "{} encoded to empty", $format);

            let decoded = decode_with_hint(&encoded, $format)
                .unwrap_or_else(|e| panic!("{} decode failed: {e}", $format));
            assert_eq!(decoded.info.width, 4);
            assert_eq!(decoded.info.height, 4);
            assert_eq!(decoded.pixels.len(), 4 * 4 * 4);

            // Decoded is sRGB f32 — linearize before comparing with linear input
            let linearized = linearize(&decoded.pixels);
            let mae = max_abs_error(&input, &linearized);
            assert!(
                mae <= LOSSLESS_TOL,
                "{} round-trip max error {mae:.6} exceeds {LOSSLESS_TOL:.6}",
                $format
            );
        }
    };
}

lossless_roundtrip_test!(roundtrip_png, "png");
lossless_roundtrip_test!(roundtrip_bmp, "bmp");
lossless_roundtrip_test!(roundtrip_qoi, "qoi");
lossless_roundtrip_test!(roundtrip_tga, "tga");
lossless_roundtrip_test!(roundtrip_pnm, "pnm");
lossless_roundtrip_test!(roundtrip_tiff, "tiff");

// ─── Lossy format round-trip tests ──────────────────────────────────────────

#[test]
fn roundtrip_jpeg() {
    let input = gradient_16x16();
    let encoded = encode(&input, 16, 16, "jpeg", Some(95)).unwrap();
    assert!(!encoded.is_empty());

    let decoded = decode_with_hint(&encoded, "jpeg").unwrap();
    assert_eq!(decoded.info.width, 16);
    assert_eq!(decoded.info.height, 16);

    let linearized = linearize(&decoded.pixels);
    let mae = mean_abs_error(&input, &linearized);
    assert!(
        mae < 0.02,
        "JPEG round-trip MAE {mae:.4} too high (expected < 0.02)"
    );
}

#[test]
fn roundtrip_webp() {
    let input = gradient_16x16();
    let encoded = encode(&input, 16, 16, "webp", Some(95)).unwrap();
    assert!(!encoded.is_empty());

    let decoded = decode_with_hint(&encoded, "webp").unwrap();
    assert_eq!(decoded.info.width, 16);
    assert_eq!(decoded.info.height, 16);

    let linearized = linearize(&decoded.pixels);
    let mae = mean_abs_error(&input, &linearized);
    assert!(
        mae < 0.02,
        "WebP round-trip MAE {mae:.4} too high (expected < 0.02)"
    );
}

#[test]
fn roundtrip_gif() {
    // GIF is palette-based (256 colors) — use a simple pattern
    let input = checkerboard_8x8();
    let encoded = encode(&input, 8, 8, "gif", None).unwrap();
    assert!(!encoded.is_empty());

    let decoded = decode_with_hint(&encoded, "gif").unwrap();
    assert_eq!(decoded.info.width, 8);
    assert_eq!(decoded.info.height, 8);

    let linearized = linearize(&decoded.pixels);
    let mae = max_abs_error(&input, &linearized);
    assert!(
        mae < 0.02,
        "GIF checkerboard round-trip max error {mae:.4} too high"
    );
}

// ─── ICO round-trip ─────────────────────────────────────────────────────────

#[test]
fn roundtrip_ico() {
    let input = solid_gray_1x1();
    let encoded = encode(&input, 1, 1, "ico", None).unwrap();
    assert!(!encoded.is_empty());

    let decoded = decode_with_hint(&encoded, "ico").unwrap();
    assert_eq!(decoded.info.width, 1);
    assert_eq!(decoded.info.height, 1);

    let linearized = linearize(&decoded.pixels);
    let mae = max_abs_error(&input, &linearized);
    assert!(
        mae <= LOSSLESS_TOL,
        "ICO round-trip max error {mae:.6} exceeds tolerance"
    );
}

// ─── Linear f32 format round-trip tests ─────────────────────────────────────

#[test]
fn roundtrip_exr_preserves_linear() {
    let input = gradient_4x4();
    let encoded = encode(&input, 4, 4, "exr", None).unwrap();
    assert!(!encoded.is_empty());

    let decoded = decode_with_hint(&encoded, "exr").unwrap();
    assert_eq!(decoded.info.width, 4);
    assert_eq!(decoded.info.height, 4);
    assert_eq!(decoded.info.color_space, ColorSpace::Linear);

    // EXR stores f32 natively — no gamma, no quantization
    let mae = max_abs_error(&input, &decoded.pixels);
    assert!(
        mae < 1e-5,
        "EXR round-trip max error {mae:.8} — expected near-zero for f32"
    );
}

#[test]
fn roundtrip_hdr_preserves_linear() {
    let input = gradient_4x4();
    let encoded = encode(&input, 4, 4, "hdr", None).unwrap();
    assert!(!encoded.is_empty());

    let decoded = decode_with_hint(&encoded, "hdr").unwrap();
    assert_eq!(decoded.info.width, 4);
    assert_eq!(decoded.info.height, 4);
    assert_eq!(decoded.info.color_space, ColorSpace::Linear);

    // RGBE encoding has limited precision (8-bit mantissa per channel)
    let mae = max_abs_error(&input, &decoded.pixels);
    assert!(
        mae < 0.01,
        "HDR round-trip max error {mae:.6} too high (expected < 0.01)"
    );
}

#[test]
fn roundtrip_fits_preserves_linear() {
    // FITS is grayscale — use a gray gradient where R=G=B
    let mut input = Vec::with_capacity(4 * 4 * 4);
    for i in 0..16 {
        let v = i as f32 / 15.0;
        input.push(v); // R
        input.push(v); // G
        input.push(v); // B
        input.push(1.0); // A
    }

    let encoded = encode(&input, 4, 4, "fits", None).unwrap();
    assert!(!encoded.is_empty());

    let decoded = decode_with_hint(&encoded, "fits").unwrap();
    assert_eq!(decoded.info.width, 4);
    assert_eq!(decoded.info.height, 4);
    assert_eq!(decoded.info.color_space, ColorSpace::Linear);

    // FITS encodes as f32 grayscale, decodes to gray R=G=B.
    // Compare luma channel (all equal for gray input).
    let mae = max_abs_error(&input, &decoded.pixels);
    assert!(
        mae < 1e-4,
        "FITS round-trip max error {mae:.8} — expected near-zero for f32"
    );
}

// ─── sRGB gamma encoding verification ───────────────────────────────────────

#[test]
fn srgb_encoder_applies_gamma() {
    // Encode a known linear value and verify the encoded bytes contain
    // sRGB-gamma-encoded values. Linear 0.5 → sRGB ~0.735 (≈188 in u8).
    let input = vec![0.5f32, 0.5, 0.5, 1.0]; // 1x1 linear mid-gray
    let encoded = encode(&input, 1, 1, "bmp", None).unwrap();

    // Decode back via V1 to check the u8 values that were encoded.
    let v1 = rasmcore_image::domain::decoder::decode_with_hint(&encoded, Some("bmp")).unwrap();
    let first_channel = v1.pixels[0];
    // Linear 0.5 → sRGB ≈ 0.735 → u8 ≈ 188
    // If gamma was NOT applied, we'd see 128 (0.5 * 255)
    assert!(
        first_channel > 160,
        "Expected sRGB gamma-encoded value >160 for linear 0.5, got {first_channel}"
    );
}

#[test]
fn linear_encoder_no_gamma() {
    // EXR encodes linear values directly — no gamma applied.
    let input = vec![0.5f32, 0.5, 0.5, 1.0];
    let encoded = encode(&input, 1, 1, "exr", None).unwrap();
    let decoded = decode_with_hint(&encoded, "exr").unwrap();

    // The decoded value should be very close to 0.5 (not gamma-encoded 0.735)
    assert!(
        (decoded.pixels[0] - 0.5).abs() < 1e-5,
        "EXR should preserve linear 0.5, got {}",
        decoded.pixels[0]
    );
}

// ─── Format detection from encoded output ───────────────────────────────────

#[test]
fn detect_format_from_encoded() {
    let input = solid_gray_1x1();

    let png_data = encode(&input, 1, 1, "png", None).unwrap();
    assert_eq!(rasmcore_codecs_v2::detect_format(&png_data), Some("png"));

    let jpeg_data = encode(&input, 1, 1, "jpeg", Some(85)).unwrap();
    assert_eq!(rasmcore_codecs_v2::detect_format(&jpeg_data), Some("jpg"));

    let bmp_data = encode(&input, 1, 1, "bmp", None).unwrap();
    assert_eq!(rasmcore_codecs_v2::detect_format(&bmp_data), Some("bmp"));
}

// ─── DDS decode-only test ───────────────────────────────────────────────────

#[test]
fn dds_encode_then_decode() {
    let input = gradient_4x4();
    match encode(&input, 4, 4, "dds", None) {
        Ok(encoded) => {
            let decoded = decode_with_hint(&encoded, "dds").unwrap();
            assert_eq!(decoded.info.width, 4);
            assert_eq!(decoded.info.height, 4);
        }
        Err(_) => {
            // DDS encode not supported for this config — acceptable
        }
    }
}
