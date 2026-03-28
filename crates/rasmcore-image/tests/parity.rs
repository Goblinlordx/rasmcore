/// Parity tests for rasmcore-image domain functions.
///
/// Compares rasmcore output against ImageMagick reference outputs.
/// Uses SSIM-like comparison for lossy operations, pixel-exact for lossless.
///
/// Prerequisites: run `tests/fixtures/generate.sh` first.
use std::path::Path;

use rasmcore_image::domain::types::*;
use rasmcore_image::domain::{decoder, encoder, filters, transform};

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/generated")
}

fn load_fixture(name: &str) -> Vec<u8> {
    let path = fixtures_dir().join("inputs").join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read fixture {}: {}. Run tests/fixtures/generate.sh first.",
            path.display(),
            e
        )
    })
}

fn load_reference(name: &str) -> Vec<u8> {
    let path = fixtures_dir().join("reference").join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read reference {}: {}. Run tests/fixtures/generate.sh first.",
            path.display(),
            e
        )
    })
}

/// Calculate mean absolute error between two pixel buffers
fn mean_absolute_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "pixel buffer length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum();
    sum / a.len() as f64
}

/// Calculate peak signal-to-noise ratio
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as f64 - y as f64;
            diff * diff
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

// =============================================================================
// Decode parity
// =============================================================================

#[test]
fn parity_decode_png_format_detection() {
    let data = load_fixture("gradient_64x64.png");
    assert_eq!(decoder::detect_format(&data), Some("png".to_string()));
}

#[test]
fn parity_decode_jpeg_format_detection() {
    let data = load_fixture("gradient_64x64.jpeg");
    assert_eq!(decoder::detect_format(&data), Some("jpeg".to_string()));
}

#[test]
fn parity_decode_png_dimensions() {
    let data = load_fixture("gradient_64x64.png");
    let result = decoder::decode(&data).unwrap();
    assert_eq!(result.info.width, 64);
    assert_eq!(result.info.height, 64);
}

#[test]
fn parity_decode_all_formats() {
    for (name, expected_format) in [
        ("gradient_64x64.png", Some("png")),
        ("gradient_64x64.jpeg", Some("jpeg")),
        ("gradient_64x64.webp", Some("webp")),
        ("gradient_64x64.gif", Some("gif")),
        ("gradient_64x64.bmp", Some("bmp")),
        ("gradient_64x64.tiff", Some("tiff")),
        ("gradient_64x64.qoi", Some("qoi")),
    ] {
        let data = load_fixture(name);
        let detected = decoder::detect_format(&data);
        assert_eq!(
            detected.as_deref(),
            expected_format,
            "format detection failed for {name}"
        );
        let result = decoder::decode(&data);
        assert!(
            result.is_ok(),
            "decode failed for {name}: {:?}",
            result.err()
        );
        let img = result.unwrap();
        assert_eq!(img.info.width, 64, "width mismatch for {name}");
        assert_eq!(img.info.height, 64, "height mismatch for {name}");
    }
}

// =============================================================================
// Encode roundtrip parity
// =============================================================================

#[test]
fn parity_encode_png_roundtrip_exact() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();
    let encoded = encoder::encode(&decoded.pixels, &decoded.info, "png", None).unwrap();
    let re_decoded = decoder::decode(&encoded).unwrap();
    // PNG is lossless — pixels must match exactly
    assert_eq!(decoded.pixels, re_decoded.pixels);
}

#[test]
fn parity_encode_jpeg_roundtrip_quality() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();
    let encoded = encoder::encode(&decoded.pixels, &decoded.info, "jpeg", Some(95)).unwrap();
    // JPEG is lossy — check PSNR > 30dB (generous for high quality)
    let decoded_rgba = decoder::decode_as(&data, PixelFormat::Rgba8).unwrap();
    let re_decoded_rgba = decoder::decode_as(&encoded, PixelFormat::Rgba8).unwrap();
    let quality = psnr(&decoded_rgba.pixels, &re_decoded_rgba.pixels);
    assert!(
        quality > 30.0,
        "JPEG roundtrip PSNR too low: {quality:.1}dB"
    );
}

// =============================================================================
// Transform parity (vs ImageMagick reference)
// =============================================================================

#[test]
fn parity_resize_lanczos() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();

    let resized = transform::resize(
        &decoded.pixels,
        &decoded.info,
        32,
        16,
        ResizeFilter::Lanczos3,
    )
    .unwrap();
    assert_eq!(resized.info.width, 32);
    assert_eq!(resized.info.height, 16);

    // Compare against ImageMagick reference
    let ref_data = load_reference("resize_lanczos_32x16.png");
    let ref_decoded = decoder::decode(&ref_data).unwrap();

    // Resize algorithms differ between libraries — allow MAE < 10
    let mae = mean_absolute_error(&resized.pixels, &ref_decoded.pixels);
    assert!(mae < 10.0, "resize MAE vs ImageMagick too high: {mae:.2}");
}

#[test]
fn parity_crop() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();

    let cropped = transform::crop(&decoded.pixels, &decoded.info, 8, 8, 16, 16).unwrap();

    let ref_data = load_reference("crop_16x16_8_8.png");
    let ref_decoded = decoder::decode(&ref_data).unwrap();

    // Crop should be pixel-exact
    assert_eq!(cropped.info.width, ref_decoded.info.width);
    assert_eq!(cropped.info.height, ref_decoded.info.height);

    let mae = mean_absolute_error(&cropped.pixels, &ref_decoded.pixels);
    assert!(mae < 1.0, "crop MAE should be near-zero: {mae:.2}");
}

#[test]
fn parity_rotate_90() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();

    let rotated = transform::rotate(&decoded.pixels, &decoded.info, Rotation::R90).unwrap();

    let ref_data = load_reference("rotate_90.png");
    let ref_decoded = decoder::decode(&ref_data).unwrap();

    assert_eq!(rotated.info.width, ref_decoded.info.width);
    assert_eq!(rotated.info.height, ref_decoded.info.height);

    let mae = mean_absolute_error(&rotated.pixels, &ref_decoded.pixels);
    assert!(mae < 1.0, "rotate 90 MAE should be near-zero: {mae:.2}");
}

#[test]
fn parity_rotate_180() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();
    let rotated = transform::rotate(&decoded.pixels, &decoded.info, Rotation::R180).unwrap();

    let ref_data = load_reference("rotate_180.png");
    let ref_decoded = decoder::decode(&ref_data).unwrap();

    let mae = mean_absolute_error(&rotated.pixels, &ref_decoded.pixels);
    assert!(mae < 1.0, "rotate 180 MAE: {mae:.2}");
}

#[test]
fn parity_flip_horizontal() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();
    let flipped =
        transform::flip(&decoded.pixels, &decoded.info, FlipDirection::Horizontal).unwrap();

    let ref_data = load_reference("flip_horizontal.png");
    let ref_decoded = decoder::decode(&ref_data).unwrap();

    let mae = mean_absolute_error(&flipped.pixels, &ref_decoded.pixels);
    assert!(mae < 1.0, "flip horizontal MAE: {mae:.2}");
}

#[test]
fn parity_flip_vertical() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();
    let flipped = transform::flip(&decoded.pixels, &decoded.info, FlipDirection::Vertical).unwrap();

    let ref_data = load_reference("flip_vertical.png");
    let ref_decoded = decoder::decode(&ref_data).unwrap();

    let mae = mean_absolute_error(&flipped.pixels, &ref_decoded.pixels);
    assert!(mae < 1.0, "flip vertical MAE: {mae:.2}");
}

#[test]
fn parity_grayscale() {
    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder::decode(&data).unwrap();
    let gray = filters::grayscale(&decoded.pixels, &decoded.info).unwrap();

    let ref_data = load_reference("grayscale.png");
    // Decode reference as Gray8 to match our output format
    let ref_decoded = decoder::decode_as(&ref_data, PixelFormat::Gray8).unwrap();

    assert_eq!(gray.info.format, PixelFormat::Gray8);
    assert_eq!(gray.info.width, ref_decoded.info.width);
    assert_eq!(gray.info.height, ref_decoded.info.height);

    // Compare gray channels — allow small difference due to color space math
    let mae = mean_absolute_error(&gray.pixels, &ref_decoded.pixels);
    assert!(mae < 5.0, "grayscale MAE: {mae:.2}");
}

// =============================================================================
// PNG encode parity (compression curve + pixel-exact + file size)
// =============================================================================

#[test]
fn parity_png_encode_determinism() {
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();

    let config = encoder::png::PngEncodeConfig::default();
    let img = encoder::pixels_to_dynamic_image(&decoded.pixels, &decoded.info).unwrap();
    let result1 = encoder::png::encode(&img, &decoded.info, &config).unwrap();
    let result2 = encoder::png::encode(&img, &decoded.info, &config).unwrap();
    assert_eq!(result1, result2, "PNG encode must be deterministic (byte-identical)");
}

#[test]
fn parity_png_encode_filter_size_variation() {
    // The image crate uses fdeflate (Fast) which is both faster and produces better
    // compression than flate2 for most images. Compression level maps to fdeflate
    // unconditionally. Filter type provides the meaningful size variation.
    use encoder::png::PngFilterType;
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();
    let img = encoder::pixels_to_dynamic_image(&decoded.pixels, &decoded.info).unwrap();

    let filters = [
        ("NoFilter", PngFilterType::NoFilter),
        ("Sub", PngFilterType::Sub),
        ("Up", PngFilterType::Up),
        ("Avg", PngFilterType::Avg),
        ("Paeth", PngFilterType::Paeth),
        ("Adaptive", PngFilterType::Adaptive),
    ];

    let mut sizes: Vec<(&str, usize)> = Vec::new();
    for (name, filter) in &filters {
        let config = encoder::png::PngEncodeConfig {
            compression_level: 6,
            filter_type: *filter,
        };
        let encoded = encoder::png::encode(&img, &decoded.info, &config).unwrap();
        sizes.push((name, encoded.len()));
    }

    eprintln!("PNG filter size variation (rasmcore, compression 6):");
    for (name, size) in &sizes {
        eprintln!("  {name}: {size} bytes");
    }

    // Adaptive should be among the smallest (within 5% of best)
    let adaptive_size = sizes.iter().find(|(n, _)| *n == "Adaptive").unwrap().1;
    let min_size = sizes.iter().map(|(_, s)| *s).min().unwrap();
    let ratio = adaptive_size as f64 / min_size as f64;
    assert!(
        ratio <= 1.05,
        "Adaptive ({adaptive_size}) should be within 5% of best filter ({min_size}), ratio={ratio:.3}",
    );

    // Different filters should produce different sizes (not all identical)
    let unique_sizes: std::collections::HashSet<usize> = sizes.iter().map(|(_, s)| *s).collect();
    assert!(
        unique_sizes.len() >= 2,
        "at least 2 distinct sizes expected across filter types, got {}",
        unique_sizes.len(),
    );
}

#[test]
fn parity_png_encode_all_compressions_pixel_exact() {
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();
    let img = encoder::pixels_to_dynamic_image(&decoded.pixels, &decoded.info).unwrap();

    for level in [0u8, 3, 6, 9] {
        let config = encoder::png::PngEncodeConfig {
            compression_level: level,
            filter_type: encoder::png::PngFilterType::Adaptive,
        };
        let encoded = encoder::png::encode(&img, &decoded.info, &config).unwrap();
        let re_decoded = decoder::decode(&encoded).unwrap();
        assert_eq!(
            re_decoded.pixels, decoded.pixels,
            "PNG roundtrip at compression {level} must be pixel-exact"
        );
    }
}

#[test]
fn parity_png_encode_all_filters_pixel_exact() {
    use encoder::png::PngFilterType;
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();
    let img = encoder::pixels_to_dynamic_image(&decoded.pixels, &decoded.info).unwrap();

    let filters = [
        PngFilterType::NoFilter,
        PngFilterType::Sub,
        PngFilterType::Up,
        PngFilterType::Avg,
        PngFilterType::Paeth,
        PngFilterType::Adaptive,
    ];

    for filter in filters {
        let config = encoder::png::PngEncodeConfig {
            compression_level: 6,
            filter_type: filter,
        };
        let encoded = encoder::png::encode(&img, &decoded.info, &config).unwrap();
        let re_decoded = decoder::decode(&encoded).unwrap();
        assert_eq!(
            re_decoded.pixels, decoded.pixels,
            "PNG roundtrip with filter {filter:?} must be pixel-exact"
        );
    }
}

#[test]
fn parity_png_encode_vs_imagemagick_pixel_exact() {
    // Both rasmcore and ImageMagick produce lossless PNG —
    // decoded pixels MUST be identical regardless of compression settings.
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();
    let img = encoder::pixels_to_dynamic_image(&decoded.pixels, &decoded.info).unwrap();

    for (level, ref_name) in [
        (0, "png_compress_0.png"),
        (3, "png_compress_3.png"),
        (6, "png_compress_6.png"),
        (9, "png_compress_9.png"),
    ] {
        let ref_data = load_reference(ref_name);
        let ref_decoded = decoder::decode(&ref_data).unwrap();

        // Lossless: decoded pixels must match
        assert_eq!(
            decoded.pixels, ref_decoded.pixels,
            "PNG at compression {level}: rasmcore and ImageMagick decoded pixels must be identical"
        );

        // Compare file sizes (informational — both should be in same ballpark)
        let config = encoder::png::PngEncodeConfig {
            compression_level: level,
            filter_type: encoder::png::PngFilterType::Adaptive,
        };
        let our_encoded = encoder::png::encode(&img, &decoded.info, &config).unwrap();
        let ratio = our_encoded.len() as f64 / ref_data.len() as f64;
        eprintln!(
            "PNG compression {level}: rasmcore={} bytes, ImageMagick={} bytes, ratio={ratio:.2}x",
            our_encoded.len(),
            ref_data.len(),
        );

        // File size should be within 1.5x of ImageMagick
        assert!(
            ratio <= 1.5,
            "PNG at compression {level}: rasmcore ({} bytes) is {ratio:.2}x of ImageMagick ({} bytes) — exceeds 1.5x threshold",
            our_encoded.len(),
            ref_data.len(),
        );
    }
}
