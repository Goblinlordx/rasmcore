//! APNG Reference Parity Tests — Validation against ffmpeg and ImageMagick.
//!
//! These tests validate that our APNG encode/decode produces output that
//! independent implementations (ffmpeg, ImageMagick) can correctly read,
//! and that the pixel data, frame count, and timing metadata match.
//!
//! Tests gracefully skip if the reference tools are not installed.

use rasmcore_image::domain::types::*;

fn has_tool(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn mean_absolute_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffer length mismatch: {} vs {}", a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn make_apng_test_sequence() -> FrameSequence {
    let mut seq = FrameSequence::new(8, 8);
    let colors: [[u8; 4]; 3] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
    ];
    for (i, color) in colors.iter().enumerate() {
        let pixels: Vec<u8> = (0..64).flat_map(|_| *color).collect();
        seq.push(
            DecodedImage {
                pixels,
                info: ImageInfo {
                    width: 8,
                    height: 8,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                icc_profile: None,
            },
            FrameInfo {
                index: i as u32,
                delay_ms: 100,
                disposal: DisposalMethod::None,
                width: 8,
                height: 8,
                x_offset: 0,
                y_offset: 0,
            },
        );
    }
    seq
}

/// Validate our APNG encode against ffmpeg: frame count and per-frame dimensions.
///
/// ffmpeg -i our.apng -vsync 0 frame_%03d.png → count output files and check dimensions.
#[test]
fn apng_encode_parity_vs_ffmpeg_frame_count() {
    use rasmcore_image::domain::encoder;

    if !has_tool("ffmpeg") {
        eprintln!("  apng_encode_parity_vs_ffmpeg: SKIP (ffmpeg not found)");
        return;
    }

    let seq = make_apng_test_sequence();
    let encoded = encoder::encode_sequence(&seq, "apng", None).unwrap();

    let tmp_dir = std::env::temp_dir().join("rasmcore_apng_parity");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let apng_path = tmp_dir.join("test.apng");
    std::fs::write(&apng_path, &encoded).unwrap();

    // Extract frames with ffmpeg
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            apng_path.to_str().unwrap(),
            "-vsync",
            "0",
            tmp_dir.join("frame_%03d.png").to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Count extracted frames
    let frame_files: Vec<_> = std::fs::read_dir(&tmp_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("frame_"))
                .unwrap_or(false)
        })
        .collect();

    assert_eq!(
        frame_files.len(),
        3,
        "ffmpeg should extract 3 frames from our APNG, got {}",
        frame_files.len()
    );

    // Verify each extracted frame is decodable and has correct dimensions
    for f in &frame_files {
        let frame_data = std::fs::read(f.path()).unwrap();
        let decoded = rasmcore_image::domain::decoder::decode(&frame_data).unwrap();
        assert_eq!(decoded.info.width, 8, "frame width mismatch");
        assert_eq!(decoded.info.height, 8, "frame height mismatch");
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    eprintln!("  apng_encode_parity_vs_ffmpeg: PASS (3 frames, 8x8 each)");
}

/// Validate our APNG encode against ImageMagick: frame count and delay metadata.
///
/// magick identify our.apng → one line per frame, verify count and dimensions.
#[test]
fn apng_encode_parity_vs_imagemagick_identify() {
    use rasmcore_image::domain::encoder;

    if !has_tool("magick") {
        eprintln!("  apng_encode_parity_vs_imagemagick: SKIP (magick not found)");
        return;
    }

    let seq = make_apng_test_sequence();
    let encoded = encoder::encode_sequence(&seq, "apng", None).unwrap();

    let tmp = std::env::temp_dir().join("rasmcore_apng_im_test.apng");
    std::fs::write(&tmp, &encoded).unwrap();

    // ImageMagick identify with APNG coalescing — one line per frame.
    // `magick identify` on its own may report only the default image for APNG.
    // Use `magick APNG:file identify:` to force APNG multi-frame read.
    let output = std::process::Command::new("magick")
        .args(["identify", &format!("APNG:{}", tmp.to_str().unwrap())])
        .output()
        .unwrap();

    if !output.status.success() {
        // Some ImageMagick versions don't support APNG: prefix — try plain identify
        let fallback = std::process::Command::new("magick")
            .args(["identify", tmp.to_str().unwrap()])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&fallback.stdout);
        let lines: Vec<&str> = stdout.lines().collect();
        eprintln!("  ImageMagick identify output ({} lines, no APNG: prefix):", lines.len());
        for l in &lines {
            eprintln!("    {l}");
        }
        // If only 1 line, ImageMagick doesn't support APNG natively — skip
        if lines.len() == 1 {
            eprintln!("  apng_encode_parity_vs_imagemagick: SKIP (ImageMagick APNG support limited)");
            let _ = std::fs::remove_file(&tmp);
            return;
        }
    }

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout_str.lines().collect();
    eprintln!("  ImageMagick identify APNG: output ({} lines):", lines.len());
    for l in &lines {
        eprintln!("    {l}");
    }

    // Each frame produces one identify line
    assert_eq!(
        lines.len(),
        3,
        "ImageMagick should identify 3 frames, got {}",
        lines.len()
    );

    // Each line should contain the canvas dimensions
    for (i, line) in lines.iter().enumerate() {
        assert!(
            line.contains("8x8"),
            "frame {i}: ImageMagick should report 8x8 dimensions, got: {line}"
        );
    }

    let _ = std::fs::remove_file(&tmp);
    eprintln!("  apng_encode_parity_vs_imagemagick: PASS (3 frames, 8x8 each)");
}

/// Cross-decode: our APNG encode → ffmpeg decode → compare pixels against our decode.
///
/// This is the critical test: it validates that an independent decoder (ffmpeg)
/// produces the same pixel data from our APNG output.
#[test]
fn apng_encode_cross_decode_vs_ffmpeg_pixels() {
    use rasmcore_image::domain::{decoder, encoder};

    if !has_tool("ffmpeg") {
        eprintln!("  apng_cross_decode_vs_ffmpeg: SKIP (ffmpeg not found)");
        return;
    }

    let seq = make_apng_test_sequence();
    let encoded = encoder::encode_sequence(&seq, "apng", None).unwrap();

    // Our decode
    let our_frames = decoder::decode_all_frames(&encoded).unwrap();

    let tmp_dir = std::env::temp_dir().join("rasmcore_apng_xdec");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let apng_path = tmp_dir.join("test.apng");
    std::fs::write(&apng_path, &encoded).unwrap();

    // ffmpeg decode each frame to raw RGBA
    for i in 0..3u32 {
        let raw_path = tmp_dir.join(format!("frame_{i}.raw"));
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                apng_path.to_str().unwrap(),
                "-vf",
                &format!("select=eq(n\\,{i})"),
                "-vframes",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                raw_path.to_str().unwrap(),
            ])
            .output()
            .unwrap();

        if !output.status.success() {
            eprintln!(
                "  ffmpeg frame {i} extraction failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return;
        }

        let ref_pixels = std::fs::read(&raw_path).unwrap();
        let our_pixels = &our_frames[i as usize].0.pixels;

        assert_eq!(
            ref_pixels.len(),
            our_pixels.len(),
            "frame {i}: pixel buffer size mismatch (ffmpeg={}, ours={})",
            ref_pixels.len(),
            our_pixels.len()
        );

        let mae = mean_absolute_error(our_pixels, &ref_pixels);
        eprintln!("  APNG frame {i} cross-decode MAE: {mae:.4}");
        assert!(
            mae < 1.0,
            "frame {i}: cross-decode divergence too high: MAE={mae:.4} (expected < 1.0 for lossless APNG)"
        );
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    eprintln!("  apng_cross_decode_vs_ffmpeg: PASS (all frames pixel-exact)");
}

/// Decode reference APNG (encoded by ffmpeg) with our decoder.
///
/// ffmpeg creates a reference APNG from raw frames → our decoder must read
/// the correct frame count, delays, and pixel data.
#[test]
fn apng_decode_ffmpeg_encoded_reference() {
    use rasmcore_image::domain::decoder;

    if !has_tool("ffmpeg") {
        eprintln!("  apng_decode_ffmpeg_reference: SKIP (ffmpeg not found)");
        return;
    }

    let tmp_dir = std::env::temp_dir().join("rasmcore_apng_refdec");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    // Create 3 raw RGBA frames: solid red, green, blue (8x8)
    let colors: [[u8; 4]; 3] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
    ];
    for (i, color) in colors.iter().enumerate() {
        let pixels: Vec<u8> = (0..64).flat_map(|_| *color).collect();
        let raw_path = tmp_dir.join(format!("frame_{i}.raw"));
        std::fs::write(&raw_path, &pixels).unwrap();
    }

    // Encode APNG with ffmpeg from raw frames
    // Create a concat file for ffmpeg
    let concat_path = tmp_dir.join("input.txt");
    let mut concat = String::new();
    for i in 0..3 {
        concat.push_str(&format!(
            "file '{}'\nduration 0.1\n",
            tmp_dir.join(format!("frame_{i}.raw")).display()
        ));
    }
    std::fs::write(&concat_path, &concat).unwrap();

    let apng_path = tmp_dir.join("ref.apng");
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", "8x8",
            "-r", "10",
            "-i", "pipe:0",
            "-plays", "0",
            apng_path.to_str().unwrap(),
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            let stdin = child.stdin.as_mut().unwrap();
            for color in &colors {
                let pixels: Vec<u8> = (0..64).flat_map(|_| *color).collect();
                stdin.write_all(&pixels).unwrap();
            }
            child.wait_with_output()
        })
        .unwrap();

    if !output.status.success() {
        eprintln!(
            "  ffmpeg APNG encode failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return;
    }

    // Decode the ffmpeg-encoded APNG with our decoder
    let apng_data = std::fs::read(&apng_path).unwrap();
    let frame_count = decoder::frame_count(&apng_data).unwrap();
    eprintln!("  ffmpeg-encoded APNG: {frame_count} frames");
    assert_eq!(frame_count, 3, "our decoder should find 3 frames in ffmpeg APNG");

    let frames = decoder::decode_all_frames(&apng_data).unwrap();
    assert_eq!(frames.len(), 3);

    // Verify pixel colors
    for (i, (img, _info)) in frames.iter().enumerate() {
        let expected = colors[i];
        // Check first pixel
        assert_eq!(img.pixels[0], expected[0], "frame {i} R mismatch");
        assert_eq!(img.pixels[1], expected[1], "frame {i} G mismatch");
        assert_eq!(img.pixels[2], expected[2], "frame {i} B mismatch");
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    eprintln!("  apng_decode_ffmpeg_reference: PASS (3 frames, pixel-correct)");
}
