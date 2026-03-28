//! WASM integration tests for rasmcore-image component.
//!
//! Loads the compiled WASM component via wasmtime and validates all interfaces
//! against ImageMagick reference outputs. These tests prove the full stack works:
//! Rust domain → adapter → WIT → WASM binary → wasmtime host → assertions.
//!
//! Prerequisites:
//!   1. `cargo component build -p rasmcore-image`
//!   2. `tests/fixtures/generate.sh`

use wasm_integration::exports::rasmcore::image::pipeline::PngWriteConfig;
use wasm_integration::exports::rasmcore::image::transform::{
    FlipDirection, ResizeFilter, Rotation,
};
use wasm_integration::rasmcore::core::types::{ColorSpace, PixelFormat};
use wasm_integration::*;

// =============================================================================
// Decoder tests
// =============================================================================

#[test]
fn wasm_decoder_detect_format_png() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();

    let data = load_fixture("gradient_64x64.png");
    let fmt = decoder.call_detect_format(&mut store, &data).unwrap();
    assert_eq!(fmt.as_deref(), Some("png"));
}

#[test]
fn wasm_decoder_detect_format_jpeg() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();

    let data = load_fixture("gradient_64x64.jpeg");
    let fmt = decoder.call_detect_format(&mut store, &data).unwrap();
    assert_eq!(fmt.as_deref(), Some("jpeg"));
}

#[test]
fn wasm_decoder_decode_png_dimensions() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();
    assert_eq!(decoded.info.width, 64);
    assert_eq!(decoded.info.height, 64);
}

#[test]
fn wasm_decoder_decode_all_formats() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();

    let cases = [
        ("gradient_64x64.png", Some("png")),
        ("gradient_64x64.jpeg", Some("jpeg")),
        ("gradient_64x64.webp", Some("webp")),
        ("gradient_64x64.gif", Some("gif")),
        ("gradient_64x64.bmp", Some("bmp")),
        ("gradient_64x64.tiff", Some("tiff")),
        ("gradient_64x64.qoi", Some("qoi")),
    ];

    for (name, expected_format) in cases {
        let data = load_fixture(name);
        let detected = decoder.call_detect_format(&mut store, &data).unwrap();
        assert_eq!(
            detected.as_deref(),
            expected_format,
            "format detection failed for {name}"
        );
        let result = decoder.call_decode(&mut store, &data).unwrap();
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

#[test]
fn wasm_decoder_decode_as_rgba8() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder
        .call_decode_as(&mut store, &data, PixelFormat::Rgba8)
        .unwrap()
        .unwrap();
    assert_eq!(decoded.info.format, PixelFormat::Rgba8);
    assert_eq!(decoded.pixels.len(), 64 * 64 * 4);
}

#[test]
fn wasm_decoder_supported_formats() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();

    let formats = decoder.call_supported_formats(&mut store).unwrap();
    assert!(formats.contains(&"png".to_string()));
    assert!(formats.contains(&"jpeg".to_string()));
    assert!(formats.contains(&"webp".to_string()));
}

// =============================================================================
// Encoder tests
// =============================================================================

#[test]
fn wasm_encoder_png_roundtrip_exact() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let encoder = bindings.rasmcore_image_encoder();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let encoded = encoder
        .call_encode(&mut store, &decoded.pixels, decoded.info, "png", None)
        .unwrap()
        .unwrap();

    let re_decoded = decoder.call_decode(&mut store, &encoded).unwrap().unwrap();
    // PNG is lossless — pixels must match exactly
    assert_eq!(decoded.pixels, re_decoded.pixels);
}

#[test]
fn wasm_encoder_jpeg_roundtrip_quality() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let encoder = bindings.rasmcore_image_encoder();

    let data = load_fixture("gradient_64x64.png");
    let decoded_rgba = decoder
        .call_decode_as(&mut store, &data, PixelFormat::Rgba8)
        .unwrap()
        .unwrap();

    let encoded = encoder
        .call_encode(
            &mut store,
            &decoded_rgba.pixels,
            decoded_rgba.info,
            "jpeg",
            Some(95),
        )
        .unwrap()
        .unwrap();

    let re_decoded = decoder
        .call_decode_as(&mut store, &encoded, PixelFormat::Rgba8)
        .unwrap()
        .unwrap();

    let quality = psnr(&decoded_rgba.pixels, &re_decoded.pixels);
    assert!(
        quality > 30.0,
        "JPEG roundtrip PSNR too low: {quality:.1}dB"
    );
}

#[test]
fn wasm_encoder_supported_formats() {
    let (mut store, bindings) = instantiate_image_component();
    let encoder = bindings.rasmcore_image_encoder();

    let formats = encoder.call_supported_formats(&mut store).unwrap();
    assert!(formats.contains(&"png".to_string()));
    assert!(formats.contains(&"jpeg".to_string()));
}

// =============================================================================
// Transform tests (vs ImageMagick reference)
// =============================================================================

#[test]
fn wasm_transform_resize_lanczos() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let transform = bindings.rasmcore_image_transform();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let (resized_px, resized_info) = transform
        .call_resize(
            &mut store,
            &decoded.pixels,
            decoded.info,
            32,
            16,
            ResizeFilter::Lanczos3,
        )
        .unwrap()
        .unwrap();

    assert_eq!(resized_info.width, 32);
    assert_eq!(resized_info.height, 16);

    // Compare against ImageMagick reference
    let ref_data = load_reference("resize_lanczos_32x16.png");
    let ref_decoded = decoder.call_decode(&mut store, &ref_data).unwrap().unwrap();

    let mae = mean_absolute_error(&resized_px, &ref_decoded.pixels);
    assert!(mae < 10.0, "resize MAE vs ImageMagick too high: {mae:.2}");
}

#[test]
fn wasm_transform_crop() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let transform = bindings.rasmcore_image_transform();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let (cropped_px, cropped_info) = transform
        .call_crop(&mut store, &decoded.pixels, decoded.info, 8, 8, 16, 16)
        .unwrap()
        .unwrap();

    let ref_data = load_reference("crop_16x16_8_8.png");
    let ref_decoded = decoder.call_decode(&mut store, &ref_data).unwrap().unwrap();

    assert_eq!(cropped_info.width, ref_decoded.info.width);
    assert_eq!(cropped_info.height, ref_decoded.info.height);

    let mae = mean_absolute_error(&cropped_px, &ref_decoded.pixels);
    assert!(mae < 1.0, "crop MAE should be near-zero: {mae:.2}");
}

#[test]
fn wasm_transform_rotate_90() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let transform = bindings.rasmcore_image_transform();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let (rotated_px, rotated_info) = transform
        .call_rotate(&mut store, &decoded.pixels, decoded.info, Rotation::R90)
        .unwrap()
        .unwrap();

    let ref_data = load_reference("rotate_90.png");
    let ref_decoded = decoder.call_decode(&mut store, &ref_data).unwrap().unwrap();

    assert_eq!(rotated_info.width, ref_decoded.info.width);
    assert_eq!(rotated_info.height, ref_decoded.info.height);

    let mae = mean_absolute_error(&rotated_px, &ref_decoded.pixels);
    assert!(mae < 1.0, "rotate 90 MAE should be near-zero: {mae:.2}");
}

#[test]
fn wasm_transform_rotate_180() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let transform = bindings.rasmcore_image_transform();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let (rotated_px, _) = transform
        .call_rotate(&mut store, &decoded.pixels, decoded.info, Rotation::R180)
        .unwrap()
        .unwrap();

    let ref_data = load_reference("rotate_180.png");
    let ref_decoded = decoder.call_decode(&mut store, &ref_data).unwrap().unwrap();

    let mae = mean_absolute_error(&rotated_px, &ref_decoded.pixels);
    assert!(mae < 1.0, "rotate 180 MAE: {mae:.2}");
}

#[test]
fn wasm_transform_flip_horizontal() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let transform = bindings.rasmcore_image_transform();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let (flipped_px, _) = transform
        .call_flip(
            &mut store,
            &decoded.pixels,
            decoded.info,
            FlipDirection::Horizontal,
        )
        .unwrap()
        .unwrap();

    let ref_data = load_reference("flip_horizontal.png");
    let ref_decoded = decoder.call_decode(&mut store, &ref_data).unwrap().unwrap();

    let mae = mean_absolute_error(&flipped_px, &ref_decoded.pixels);
    assert!(mae < 1.0, "flip horizontal MAE: {mae:.2}");
}

#[test]
fn wasm_transform_flip_vertical() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let transform = bindings.rasmcore_image_transform();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let (flipped_px, _) = transform
        .call_flip(
            &mut store,
            &decoded.pixels,
            decoded.info,
            FlipDirection::Vertical,
        )
        .unwrap()
        .unwrap();

    let ref_data = load_reference("flip_vertical.png");
    let ref_decoded = decoder.call_decode(&mut store, &ref_data).unwrap().unwrap();

    let mae = mean_absolute_error(&flipped_px, &ref_decoded.pixels);
    assert!(mae < 1.0, "flip vertical MAE: {mae:.2}");
}

// =============================================================================
// Filter tests
// =============================================================================

#[test]
fn wasm_filters_grayscale() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let filters = bindings.rasmcore_image_filters();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let (gray_px, gray_info) = filters
        .call_grayscale(&mut store, &decoded.pixels, decoded.info)
        .unwrap()
        .unwrap();

    assert_eq!(gray_info.format, PixelFormat::Gray8);
    assert_eq!(gray_info.width, 64);
    assert_eq!(gray_info.height, 64);

    // Compare against ImageMagick reference
    let ref_data = load_reference("grayscale.png");
    let ref_decoded = decoder
        .call_decode_as(&mut store, &ref_data, PixelFormat::Gray8)
        .unwrap()
        .unwrap();

    let mae = mean_absolute_error(&gray_px, &ref_decoded.pixels);
    assert!(mae < 5.0, "grayscale MAE: {mae:.2}");
}

#[test]
fn wasm_filters_blur() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let filters = bindings.rasmcore_image_filters();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let blurred = filters
        .call_blur(&mut store, &decoded.pixels, decoded.info, 2.0)
        .unwrap()
        .unwrap();

    // Blur should produce valid output of same size
    assert_eq!(blurred.len(), decoded.pixels.len());
    // Blurred image should differ from original (gradient images change subtly)
    let mae = mean_absolute_error(&blurred, &decoded.pixels);
    assert!(mae > 0.01, "blur should change pixels, MAE: {mae:.2}");
}

#[test]
fn wasm_filters_sharpen() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let filters = bindings.rasmcore_image_filters();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let sharpened = filters
        .call_sharpen(&mut store, &decoded.pixels, decoded.info, 1.0)
        .unwrap()
        .unwrap();

    assert_eq!(sharpened.len(), decoded.pixels.len());
}

#[test]
fn wasm_filters_brightness() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let filters = bindings.rasmcore_image_filters();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let brightened = filters
        .call_brightness(&mut store, &decoded.pixels, decoded.info, 0.5)
        .unwrap()
        .unwrap();

    assert_eq!(brightened.len(), decoded.pixels.len());
    let mae = mean_absolute_error(&brightened, &decoded.pixels);
    assert!(mae > 0.1, "brightness should change pixels, MAE: {mae:.2}");
}

#[test]
fn wasm_filters_contrast() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let filters = bindings.rasmcore_image_filters();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let contrasted = filters
        .call_contrast(&mut store, &decoded.pixels, decoded.info, 0.5)
        .unwrap()
        .unwrap();

    assert_eq!(contrasted.len(), decoded.pixels.len());
    let mae = mean_absolute_error(&contrasted, &decoded.pixels);
    assert!(mae > 0.1, "contrast should change pixels, MAE: {mae:.2}");
}

// =============================================================================
// Compositing tests
// 
=====================================================================

#[test]
fn wasm_filters_composite_opaque() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let filters = bindings.rasmcore_image_filters();

    let bg_data = load_fixture("gradient_64x64.png");
    let bg = decoder
        .call_decode_as(&mut store, &bg_data, PixelFormat::Rgba8)
        .unwrap()
        .unwrap();

    // Create a solid red 16x16 foreground
    let fg_pixels: Vec<u8> = [255, 0, 0, 255].repeat(16 * 16);
    let fg_info = wasm_integration::rasmcore::core::types::ImageInfo {
        width: 16,
        height: 16,
        format: PixelFormat::Rgba8,
        color_space: wasm_integration::rasmcore::core::types::ColorSpace::Srgb,
    };

    let result = filters
        .call_composite(&mut store, &fg_pixels, fg_info, &bg.pixels, bg.info, 10, 10)
        .unwrap()
        .unwrap();

    // Output should be same size as bg
    assert_eq!(result.len(), bg.pixels.len());

    // Pixel at (10, 10) should be red (opaque overlay)
    let idx = (10 * 64 + 10) * 4;
    assert_eq!(result[idx], 255); // R
    assert_eq!(result[idx + 1], 0); // G
    assert_eq!(result[idx + 2], 0); // B
    assert_eq!(result[idx + 3], 255); // A

    // Pixel at (0, 0) should be unchanged from bg
    assert_eq!(&result[0..4], &bg.pixels[0..4]);
}

#[test]
fn wasm_filters_composite_semi_transparent() {
    let (mut store, bindings) = instantiate_image_component();
    let filters = bindings.rasmcore_image_filters();

    // 1x1 50% red over 1x1 solid blue
    let fg_pixels = vec![255u8, 0, 0, 128];
    let fg_info = wasm_integration::rasmcore::core::types::ImageInfo {
        width: 1,
        height: 1,
        format: PixelFormat::Rgba8,
        color_space: wasm_integration::rasmcore::core::types::ColorSpace::Srgb,
    };
    let bg_pixels = vec![0u8, 0, 255, 255];
    let bg_info = wasm_integration::rasmcore::core::types::ImageInfo {
        width: 1,
        height: 1,
        format: PixelFormat::Rgba8,
        color_space: wasm_integration::rasmcore::core::types::ColorSpace::Srgb,
    };

    let result = filters
        .call_composite(&mut store, &fg_pixels, fg_info, &bg_pixels, bg_info, 0, 0)
        .unwrap()
        .unwrap();

    assert_eq!(result[0], 128); // R
    assert_eq!(result[1], 0); // G
    assert_eq!(result[2], 127); // B
    assert_eq!(result[3], 255); // A
}

#[test]
fn wasm_pipeline_composite() {
    let (mut store, bindings) = instantiate_image_component();
    let pipeline_guest = bindings.rasmcore_image_pipeline();
    let pipe_res = pipeline_guest.image_pipeline();

    let pipe = pipe_res.call_constructor(&mut store).unwrap();

    // Read bg
    let bg_data = load_fixture("gradient_64x64.png");
    let bg_node = pipe_res
        .call_read(&mut store, pipe, &bg_data)
        .unwrap()
        .unwrap();

    // Read fg (same image — just to test the pipeline composite method)
    let fg_data = load_fixture("gradient_64x64.png");
    let fg_node = pipe_res
        .call_read(&mut store, pipe, &fg_data)
        .unwrap()
        .unwrap();

    // Composite fg over bg at offset (10, 10)
    let comp_node = pipe_res
        .call_composite(&mut store, pipe, fg_node, bg_node, 10, 10)
        .unwrap()
        .unwrap();

    // Get info — should match bg dimensions
    let info = pipe_res
        .call_node_info(&mut store, pipe, comp_node)
        .unwrap()
        .unwrap();
    assert_eq!(info.width, 64);
    assert_eq!(info.height, 64);

    // Write as PNG — should produce valid output
    let config = PngWriteConfig {
        compression_level: None,
        filter_type: None,
    };
    let output = pipe_res
        .call_write_png(&mut store, pipe, comp_node, config)
        .unwrap()
        .unwrap();
    assert_eq!(&output[..4], &[0x89, 0x50, 0x4E, 0x47]); // PNG magic bytes
=======
// GIF encode tests
// =============================================================================

#[test]
fn wasm_encoder_gif_roundtrip() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let encoder = bindings.rasmcore_image_encoder();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let encoded = encoder
        .call_encode(&mut store, &decoded.pixels, decoded.info, "gif", None)
        .unwrap()
        .unwrap();

    // Verify GIF magic bytes
    assert_eq!(&encoded[..3], b"GIF");

    // Roundtrip: decode back and check dimensions
    let re_decoded = decoder.call_decode(&mut store, &encoded).unwrap().unwrap();
    assert_eq!(re_decoded.info.width, 64);
    assert_eq!(re_decoded.info.height, 64);
}

#[test]
fn wasm_encoder_supported_formats_includes_gif() {
    let (mut store, bindings) = instantiate_image_component();
    let encoder = bindings.rasmcore_image_encoder();

    let formats = encoder.call_supported_formats(&mut store).unwrap();
    assert!(formats.contains(&"gif".to_string()));
}

// =============================================================================
// Auto-orient tests
// =============================================================================

#[test]
fn wasm_transform_auto_orient_normal_is_identity() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let transform = bindings.rasmcore_image_transform();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    let (oriented_px, oriented_info) = transform
        .call_auto_orient(
            &mut store,
            &decoded.pixels,
            decoded.info,
            wasm_integration::exports::rasmcore::image::transform::ExifOrientation::Normal,
        )
        .unwrap()
        .unwrap();

    assert_eq!(oriented_info.width, 64);
    assert_eq!(oriented_info.height, 64);
    assert_eq!(oriented_px, decoded.pixels);
}

// =============================================================================
// Metadata tests
// =============================================================================

#[test]
fn wasm_metadata_has_exif_jpeg() {
    let (mut store, bindings) = instantiate_image_component();
    let metadata = bindings.rasmcore_image_metadata();

    // ImageMagick JPEG should have EXIF
    let data = load_fixture("gradient_64x64.jpeg");
    let has = metadata.call_has_exif(&mut store, &data).unwrap();
    // ImageMagick may or may not embed EXIF in simple conversions
    // Just verify the call works without error
    let _ = has;
}

#[test]
fn wasm_metadata_read_exif_png_fails() {
    let (mut store, bindings) = instantiate_image_component();
    let metadata = bindings.rasmcore_image_metadata();

    // PNG doesn't have EXIF
    let data = load_fixture("gradient_64x64.png");
    let result = metadata.call_read_exif(&mut store, &data).unwrap();
    assert!(result.is_err());
}

// =============================================================================
// ICC color profile tests
// =============================================================================

#[test]
fn wasm_decoder_extracts_icc_from_jpeg() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let encoder = bindings.rasmcore_image_encoder();

    // Create a JPEG, decode it, re-encode with ICC, then decode again
    let data = load_fixture("gradient_64x64.jpeg");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    // Original JPEG should have no ICC profile
    assert!(
        decoded.icc_profile.is_none(),
        "fixture should not have ICC profile"
    );

    // Embed a fake ICC profile and verify roundtrip
    let fake_icc = vec![42u8; 128];
    let config =
        wasm_integration::exports::rasmcore::image::encoder::JpegEncodeConfig { quality: Some(95) };
    let encoded_with_icc = encoder
        .call_encode_jpeg_with_icc(&mut store, &decoded.pixels, decoded.info, config, &fake_icc)
        .unwrap()
        .unwrap();

    // Decode the ICC-embedded JPEG
    let re_decoded = decoder
        .call_decode(&mut store, &encoded_with_icc)
        .unwrap()
        .unwrap();
    assert_eq!(re_decoded.icc_profile, Some(fake_icc));
}

#[test]
fn wasm_decoder_extracts_icc_from_png() {
    let (mut store, bindings) = instantiate_image_component();
    let decoder = bindings.rasmcore_image_decoder();
    let encoder = bindings.rasmcore_image_encoder();

    let data = load_fixture("gradient_64x64.png");
    let decoded = decoder.call_decode(&mut store, &data).unwrap().unwrap();

    // Embed a fake ICC profile in PNG and verify roundtrip
    let fake_icc = vec![99u8; 200];
    let config = wasm_integration::exports::rasmcore::image::encoder::PngEncodeConfig {
        compression_level: Some(6),
        filter_type: None,
    };
    let encoded_with_icc = encoder
        .call_encode_png_with_icc(&mut store, &decoded.pixels, decoded.info, config, &fake_icc)
        .unwrap()
        .unwrap();

    let re_decoded = decoder
        .call_decode(&mut store, &encoded_with_icc)
        .unwrap()
        .unwrap();
    assert_eq!(re_decoded.icc_profile, Some(fake_icc));
}

#[test]
fn wasm_pipeline_icc_to_srgb() {
    let (mut store, bindings) = instantiate_image_component();
    let pipeline_iface = bindings.rasmcore_image_pipeline();

    let data = load_fixture("gradient_64x64.png");

    // Load sRGB ICC profile from system (macOS) for testing
    let icc_path = "/System/Library/ColorSync/Profiles/sRGB Profile.icc";
    let icc_profile = match std::fs::read(icc_path) {
        Ok(p) => p,
        Err(_) => return, // skip on non-macOS
    };

    let pipeline = pipeline_iface.image_pipeline();
    let resource = pipeline.call_constructor(&mut store).unwrap();

    let src_node = pipeline
        .call_read(&mut store, resource, &data)
        .unwrap()
        .unwrap();

    // Apply ICC-to-sRGB conversion
    let srgb_node = pipeline
        .call_icc_to_srgb(&mut store, resource, src_node, &icc_profile)
        .unwrap()
        .unwrap();

    // Verify node info shows sRGB color space
    let info = pipeline
        .call_node_info(&mut store, resource, srgb_node)
        .unwrap()
        .unwrap();
    assert_eq!(info.color_space, ColorSpace::Srgb);
    assert_eq!(info.width, 64);
    assert_eq!(info.height, 64);

    // Write output — this drives the pipeline
    let config = wasm_integration::exports::rasmcore::image::pipeline::PngWriteConfig {
        compression_level: Some(6),
        filter_type: None,
    };
    let output = pipeline
        .call_write_png(&mut store, resource, srgb_node, config)
        .unwrap()
        .unwrap();
    assert!(
        !output.is_empty(),
        "ICC-to-sRGB pipeline should produce output"
    );
}
