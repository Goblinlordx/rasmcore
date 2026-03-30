/// Parity tests for rasmcore-image domain functions.
///
/// Compares rasmcore output against ImageMagick reference outputs.
/// Uses SSIM-like comparison for lossy operations, pixel-exact for lossless.
///
/// Prerequisites: run `tests/fixtures/generate.sh` first.
use std::path::Path;

use butteraugli::{ButteraugliParams, butteraugli};
use dssim_core::Dssim;
use imgref::Img;
use rasmcore_image::domain::types::*;
use rasmcore_image::domain::{decoder, encoder, filters, transform};
use rgb::RGB8;

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
    let result1 = encoder::png::encode(&decoded.pixels, &decoded.info, &config).unwrap();
    let result2 = encoder::png::encode(&decoded.pixels, &decoded.info, &config).unwrap();
    assert_eq!(
        result1, result2,
        "PNG encode must be deterministic (byte-identical)"
    );
}

#[test]
fn parity_png_encode_filter_size_variation() {
    // The image crate uses fdeflate (Fast) which is both faster and produces better
    // compression than flate2 for most images. Compression level maps to fdeflate
    // unconditionally. Filter type provides the meaningful size variation.
    use encoder::png::PngFilterType;
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();

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
        let encoded = encoder::png::encode(&decoded.pixels, &decoded.info, &config).unwrap();
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

    for level in [0u8, 3, 6, 9] {
        let config = encoder::png::PngEncodeConfig {
            compression_level: level,
            filter_type: encoder::png::PngFilterType::Adaptive,
        };
        let encoded = encoder::png::encode(&decoded.pixels, &decoded.info, &config).unwrap();
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
        let encoded = encoder::png::encode(&decoded.pixels, &decoded.info, &config).unwrap();
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
        let our_encoded = encoder::png::encode(&decoded.pixels, &decoded.info, &config).unwrap();
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

// =============================================================================
// JPEG encoder parity (zenjpeg vs ImageMagick/libjpeg-turbo)
// =============================================================================

/// Decode JPEG and return RGB8 pixels + dimensions.
fn decode_jpeg_rgb(data: &[u8]) -> (Vec<u8>, u32, u32) {
    let decoded = decoder::decode_as(data, PixelFormat::Rgb8).unwrap();
    (decoded.pixels, decoded.info.width, decoded.info.height)
}

/// Convert RGB8 byte slice to Vec<RGB8> for butteraugli/dssim.
fn bytes_to_rgb8(pixels: &[u8]) -> Vec<RGB8> {
    pixels
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect()
}

/// Compute butteraugli score between two RGB8 images.
fn butteraugli_score(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    let img_a = Img::new(bytes_to_rgb8(a), width, height);
    let img_b = Img::new(bytes_to_rgb8(b), width, height);
    let result = butteraugli(
        img_a.as_ref(),
        img_b.as_ref(),
        &ButteraugliParams::default(),
    )
    .unwrap();
    result.score
}

/// Compute DSSIM score between two RGB8 images.
fn dssim_score(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let rgb_a = bytes_to_rgb8(a);
    let rgb_b = bytes_to_rgb8(b);
    let img_a = attr.create_image_rgb(&rgb_a, width, height).unwrap();
    let img_b = attr.create_image_rgb(&rgb_b, width, height).unwrap();
    let (val, _) = attr.compare(&img_a, img_b);
    f64::from(val)
}

#[test]
fn parity_jpeg_determinism() {
    // Encode the same input twice — output must be byte-identical.
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();
    let config = encoder::jpeg::JpegEncodeConfig {
        quality: 85,
        progressive: false,
    };
    let out1 = encoder::jpeg::encode_pixels(&decoded.pixels, &decoded.info, &config).unwrap();
    let out2 = encoder::jpeg::encode_pixels(&decoded.pixels, &decoded.info, &config).unwrap();
    assert_eq!(out1, out2, "JPEG encoding is not deterministic");
}

#[test]
fn parity_jpeg_determinism_progressive() {
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();
    let config = encoder::jpeg::JpegEncodeConfig {
        quality: 85,
        progressive: true,
    };
    let out1 = encoder::jpeg::encode_pixels(&decoded.pixels, &decoded.info, &config).unwrap();
    let out2 = encoder::jpeg::encode_pixels(&decoded.pixels, &decoded.info, &config).unwrap();
    assert_eq!(out1, out2, "Progressive JPEG encoding is not deterministic");
}

#[test]
fn parity_jpeg_quality_per_byte_butteraugli() {
    // Quality-per-byte comparison: zenjpeg (jpegli quality scale) vs ImageMagick
    // (libjpeg-turbo). Since quality scales differ between encoders, we compare
    // the Butteraugli-distance-per-kilobyte ratio. zenjpeg should achieve equal
    // or better quality for a given file size across the quality range.
    let source_data = load_fixture("photo_256x256.png");
    let source = decoder::decode_as(&source_data, PixelFormat::Rgb8).unwrap();
    let (w, h) = (source.info.width as usize, source.info.height as usize);

    let mut zen_wins = 0usize;
    let mut total = 0usize;

    for q in [10u8, 30, 50, 70, 85, 95] {
        let config = encoder::jpeg::JpegEncodeConfig {
            quality: q,
            progressive: false,
        };
        let zen_jpeg = encoder::jpeg::encode_pixels(&source.pixels, &source.info, &config).unwrap();
        let (zen_pixels, _, _) = decode_jpeg_rgb(&zen_jpeg);

        let im_jpeg = load_reference(&format!("jpeg_q{q}.jpeg"));
        let (im_pixels, _, _) = decode_jpeg_rgb(&im_jpeg);

        let zen_ba = butteraugli_score(&source.pixels, &zen_pixels, w, h);
        let im_ba = butteraugli_score(&source.pixels, &im_pixels, w, h);

        // Quality-per-byte: distortion * bytes (lower = better quality/byte)
        let zen_cost = zen_ba * zen_jpeg.len() as f64;
        let im_cost = im_ba * im_jpeg.len() as f64;

        if zen_cost <= im_cost {
            zen_wins += 1;
        }
        total += 1;

        eprintln!(
            "q{q:>2}: zen={zen_ba:.3} ({} B), im={im_ba:.3} ({} B) | cost: zen={zen_cost:.0} im={im_cost:.0} {}",
            zen_jpeg.len(),
            im_jpeg.len(),
            if zen_cost <= im_cost { "✓" } else { "✗" }
        );
    }

    // zenjpeg must win quality-per-byte on majority of quality levels
    assert!(
        zen_wins > total / 2,
        "Butteraugli quality-per-byte: zenjpeg won {zen_wins}/{total} quality levels (need majority)"
    );
}

#[test]
fn parity_jpeg_quality_per_byte_dssim() {
    let source_data = load_fixture("photo_256x256.png");
    let source = decoder::decode_as(&source_data, PixelFormat::Rgb8).unwrap();
    let (w, h) = (source.info.width as usize, source.info.height as usize);

    let mut zen_wins = 0usize;
    let mut total = 0usize;

    for q in [10u8, 30, 50, 70, 85, 95] {
        let config = encoder::jpeg::JpegEncodeConfig {
            quality: q,
            progressive: false,
        };
        let zen_jpeg = encoder::jpeg::encode_pixels(&source.pixels, &source.info, &config).unwrap();
        let (zen_pixels, _, _) = decode_jpeg_rgb(&zen_jpeg);

        let im_jpeg = load_reference(&format!("jpeg_q{q}.jpeg"));
        let (im_pixels, _, _) = decode_jpeg_rgb(&im_jpeg);

        let zen_ds = dssim_score(&source.pixels, &zen_pixels, w, h);
        let im_ds = dssim_score(&source.pixels, &im_pixels, w, h);

        let zen_cost = zen_ds * zen_jpeg.len() as f64;
        let im_cost = im_ds * im_jpeg.len() as f64;

        if zen_cost <= im_cost {
            zen_wins += 1;
        }
        total += 1;

        eprintln!(
            "q{q:>2}: zen={zen_ds:.6} ({} B), im={im_ds:.6} ({} B) | cost: zen={zen_cost:.2} im={im_cost:.2} {}",
            zen_jpeg.len(),
            im_jpeg.len(),
            if zen_cost <= im_cost { "✓" } else { "✗" }
        );
    }

    assert!(
        zen_wins > total / 2,
        "DSSIM quality-per-byte: zenjpeg won {zen_wins}/{total} quality levels (need majority)"
    );
}

#[test]
fn parity_jpeg_quality_curve_filesize() {
    // Verify that at higher quality levels (where trellis quantization shines),
    // zenjpeg produces competitive file sizes.
    let source_data = load_fixture("photo_256x256.png");
    let source = decoder::decode(&source_data).unwrap();

    for q in [10u8, 30, 50, 70, 85, 95] {
        let config = encoder::jpeg::JpegEncodeConfig {
            quality: q,
            progressive: false,
        };
        let zen_jpeg = encoder::jpeg::encode_pixels(&source.pixels, &source.info, &config).unwrap();
        let im_jpeg = load_reference(&format!("jpeg_q{q}.jpeg"));

        // Allow up to 60% larger at same quality number (quality scales differ).
        // The quality-per-byte tests above validate actual encoding efficiency.
        assert!(
            zen_jpeg.len() <= (im_jpeg.len() as f64 * 1.6) as usize,
            "File size: zenjpeg ({} bytes) much larger than ImageMagick ({} bytes) at q{q}",
            zen_jpeg.len(),
            im_jpeg.len()
        );
    }
}

#[test]
fn parity_jpeg_quality_monotonic() {
    // Higher quality must produce larger files.
    let source_data = load_fixture("photo_256x256.png");
    let source = decoder::decode(&source_data).unwrap();

    let qualities = [10u8, 30, 50, 70, 85, 95];
    let sizes: Vec<usize> = qualities
        .iter()
        .map(|&q| {
            let config = encoder::jpeg::JpegEncodeConfig {
                quality: q,
                progressive: false,
            };
            encoder::jpeg::encode_pixels(&source.pixels, &source.info, &config)
                .unwrap()
                .len()
        })
        .collect();

    for i in 1..sizes.len() {
        assert!(
            sizes[i] >= sizes[i - 1],
            "Quality curve not monotonic: q{} ({} bytes) < q{} ({} bytes)",
            qualities[i],
            sizes[i],
            qualities[i - 1],
            sizes[i - 1]
        );
    }
}

#[test]
fn parity_jpeg_progressive_structure() {
    // Progressive JPEG must contain SOS markers (multiple scans).
    let source_data = load_fixture("photo_256x256.png");
    let source = decoder::decode(&source_data).unwrap();

    let config = encoder::jpeg::JpegEncodeConfig {
        quality: 85,
        progressive: true,
    };
    let jpeg_data = encoder::jpeg::encode_pixels(&source.pixels, &source.info, &config).unwrap();

    // Count SOS (Start of Scan) markers: 0xFF 0xDA
    // Progressive JPEG has multiple scans; baseline has exactly 1.
    let sos_count = jpeg_data
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xDA)
        .count();
    assert!(
        sos_count > 1,
        "Progressive JPEG should have multiple SOS markers, found {sos_count}"
    );

    // Baseline should have exactly 1 SOS
    let baseline_config = encoder::jpeg::JpegEncodeConfig {
        quality: 85,
        progressive: false,
    };
    let baseline_data =
        encoder::jpeg::encode_pixels(&source.pixels, &source.info, &baseline_config).unwrap();
    let baseline_sos = baseline_data
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xDA)
        .count();
    assert_eq!(
        baseline_sos, 1,
        "Baseline JPEG should have exactly 1 SOS marker, found {baseline_sos}"
    );
}

#[test]
fn parity_jpeg_edge_quality_1() {
    let pixels: Vec<u8> = (0..(32 * 32 * 3)).map(|i| (i % 256) as u8).collect();
    let info = ImageInfo {
        width: 32,
        height: 32,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };
    let config = encoder::jpeg::JpegEncodeConfig {
        quality: 1,
        progressive: false,
    };
    let result = encoder::jpeg::encode_pixels(&pixels, &info, &config);
    assert!(result.is_ok(), "Quality 1 should succeed");
    assert_eq!(&result.unwrap()[..2], &[0xFF, 0xD8]);
}

#[test]
fn parity_jpeg_edge_quality_100() {
    let pixels: Vec<u8> = (0..(32 * 32 * 3)).map(|i| (i % 256) as u8).collect();
    let info = ImageInfo {
        width: 32,
        height: 32,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };
    let config = encoder::jpeg::JpegEncodeConfig {
        quality: 100,
        progressive: false,
    };
    let result = encoder::jpeg::encode_pixels(&pixels, &info, &config);
    assert!(result.is_ok(), "Quality 100 should succeed");
    assert_eq!(&result.unwrap()[..2], &[0xFF, 0xD8]);
}

#[test]
fn parity_jpeg_edge_1x1() {
    let pixels = vec![128u8, 64, 32];
    let info = ImageInfo {
        width: 1,
        height: 1,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };
    let config = encoder::jpeg::JpegEncodeConfig::default();
    let result = encoder::jpeg::encode_pixels(&pixels, &info, &config);
    assert!(result.is_ok(), "1x1 image should encode");
    assert_eq!(&result.unwrap()[..2], &[0xFF, 0xD8]);
}

#[test]
fn parity_jpeg_rgba8_input() {
    // RGBA8 input — alpha channel is ignored, JPEG produced.
    let pixels: Vec<u8> = (0..(32 * 32 * 4)).map(|i| (i % 256) as u8).collect();
    let info = ImageInfo {
        width: 32,
        height: 32,
        format: PixelFormat::Rgba8,
        color_space: ColorSpace::Srgb,
    };
    let config = encoder::jpeg::JpegEncodeConfig::default();
    let result = encoder::jpeg::encode_pixels(&pixels, &info, &config);
    assert!(result.is_ok(), "RGBA8 input should produce valid JPEG");
    assert_eq!(&result.unwrap()[..2], &[0xFF, 0xD8]);
}

// =============================================================================
// TIFF encode parity (lossless, compression curve, file size)
// =============================================================================

#[test]
fn parity_tiff_encode_determinism() {
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();

    let config = encoder::tiff::TiffEncodeConfig::default();
    let r1 = encoder::tiff::encode(&decoded.pixels, &decoded.info, &config).unwrap();
    let r2 = encoder::tiff::encode(&decoded.pixels, &decoded.info, &config).unwrap();
    assert_eq!(r1, r2, "TIFF encode must be deterministic");
}

#[test]
fn parity_tiff_encode_roundtrip_pixel_exact() {
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();

    for comp in [
        encoder::tiff::TiffCompression::None,
        encoder::tiff::TiffCompression::Lzw,
        encoder::tiff::TiffCompression::Deflate,
        encoder::tiff::TiffCompression::PackBits,
    ] {
        let config = encoder::tiff::TiffEncodeConfig { compression: comp };
        let encoded = encoder::tiff::encode(&decoded.pixels, &decoded.info, &config).unwrap();
        let re_decoded = decoder::decode(&encoded).unwrap();
        assert_eq!(
            re_decoded.pixels, decoded.pixels,
            "TIFF roundtrip with {comp:?} must be pixel-exact"
        );
    }
}

#[test]
fn parity_tiff_encode_vs_imagemagick() {
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();

    for (comp, ref_name) in [
        (encoder::tiff::TiffCompression::None, "tiff_none.tiff"),
        (encoder::tiff::TiffCompression::Lzw, "tiff_lzw.tiff"),
        (encoder::tiff::TiffCompression::Deflate, "tiff_deflate.tiff"),
    ] {
        let ref_data = load_reference(ref_name);
        let ref_decoded = decoder::decode(&ref_data).unwrap();

        // Lossless: decoded pixels must match
        assert_eq!(
            decoded.pixels, ref_decoded.pixels,
            "TIFF {comp:?}: rasmcore and ImageMagick decoded pixels must be identical"
        );

        // Compare file sizes
        let config = encoder::tiff::TiffEncodeConfig { compression: comp };
        let our_encoded = encoder::tiff::encode(&decoded.pixels, &decoded.info, &config).unwrap();
        let ratio = our_encoded.len() as f64 / ref_data.len() as f64;
        eprintln!(
            "TIFF {comp:?}: rasmcore={} bytes, ImageMagick={} bytes, ratio={ratio:.2}x",
            our_encoded.len(),
            ref_data.len(),
        );

        // File size within 1.5x (different TIFF implementations may vary due to metadata)
        assert!(
            ratio <= 1.5,
            "TIFF {comp:?}: rasmcore ({}) is {ratio:.2}x of ImageMagick ({}) — exceeds 1.5x",
            our_encoded.len(),
            ref_data.len(),
        );
    }
}
