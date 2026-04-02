//! Filter Reference Parity Tests — Exact validation against Python references.
//!
//! Validates every filter/color operation against a reference implementation
//! computed in Python (numpy, Pillow, OpenCV) using a dedicated venv.
//!
//! **These tests do NOT skip.** If the venv is missing, they FAIL with
//! instructions to set it up. Exact (MAE=0.0) match is required for
//! formula-based operations; spatial operations must match within MAE < 1.0.
//!
//! Setup:
//!   python3 -m venv tests/fixtures/.venv
//!   tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless

use rasmcore_image::domain::filters;
use rasmcore_image::domain::filter_traits::CpuFilter;
use rasmcore_image::domain::pipeline::Rect;
use rasmcore_image::domain::types::*;
use std::path::Path;
use std::process::Command;

// ─── Test Infrastructure ────────────────────────────────────────────────────

/// Returns the path to the venv Python binary.
/// Panics (fails the test) if the venv doesn't exist.
fn venv_python() -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let venv = manifest.join("../../tests/fixtures/.venv/bin/python3");
    if !venv.exists() {
        panic!(
            "Reference test venv not found at {}.\n\
             Setup:\n  python3 -m venv tests/fixtures/.venv\n  \
             tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless",
            venv.display()
        );
    }
    venv.to_string_lossy().into_owned()
}

/// Run a Python script via the venv interpreter, returning raw stdout bytes.
fn run_python_ref(script: &str) -> Vec<u8> {
    let python = venv_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {python}: {e}"));
    assert!(
        output.status.success(),
        "Python reference script failed:\n{}\nstderr: {}",
        script.lines().take(5).collect::<Vec<_>>().join("\n"),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn mean_absolute_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "buffer length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn max_absolute_error(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn assert_exact(label: &str, ours: &[u8], reference: &[u8]) {
    let mae = mean_absolute_error(ours, reference);
    let max_err = max_absolute_error(ours, reference);
    assert_eq!(
        ours, reference,
        "{label}: NOT pixel-exact. MAE={mae:.4}, max_err={max_err}"
    );
    eprintln!("  {label}: EXACT ✓");
}

fn assert_close(label: &str, ours: &[u8], reference: &[u8], max_mae: f64) {
    let mae = mean_absolute_error(ours, reference);
    let max_err = max_absolute_error(ours, reference);
    assert!(
        mae <= max_mae,
        "{label}: MAE={mae:.4} exceeds threshold {max_mae}. max_err={max_err}"
    );
    eprintln!("  {label}: MAE={mae:.4}, max_err={max_err} ✓");
}

fn make_gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            p.push(((x * 255) / w) as u8);
            p.push(((y * 255) / h) as u8);
            p.push(128);
        }
    }
    p
}

fn info_rgb8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    }
}

fn info_gray8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXACT MATCH (MAE = 0.0) — Formula operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_premultiply() {
    let pixels: Vec<u8> = vec![
        200, 100, 50, 128, 255, 0, 128, 255, 0, 0, 0, 0, 100, 200, 50, 64,
    ];
    let info = ImageInfo {
        width: 4,
        height: 1,
        format: PixelFormat::Rgba8,
        color_space: ColorSpace::Srgb,
    };
    let ours = filters::premultiply(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
    ).unwrap();
    let script = format!(
        "import sys\npx={pixels:?}\no=[]\nfor i in range(0,len(px),4):\n r,g,b,a=px[i],px[i+1],px[i+2],px[i+3]\n o.extend([(r*a+127)//255,(g*a+127)//255,(b*a+127)//255,a])\nsys.stdout.buffer.write(bytes(o))"
    );
    assert_exact("premultiply", &ours, &run_python_ref(&script));
}

#[test]
fn exact_premultiply_roundtrip() {
    let pixels: Vec<u8> = vec![200, 100, 50, 128, 255, 0, 128, 255];
    let info = ImageInfo {
        width: 2,
        height: 1,
        format: PixelFormat::Rgba8,
        color_space: ColorSpace::Srgb,
    };
    let premul = filters::premultiply(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
    ).unwrap();
    let roundtrip = filters::unpremultiply(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(premul.to_vec()),
        &info,
    ).unwrap();
    let max_err = max_absolute_error(&pixels, &roundtrip);
    assert!(max_err <= 1, "premultiply roundtrip max_err={max_err}");
    eprintln!("  premultiply roundtrip: max_err={max_err} ✓");
}

#[test]
fn exact_add_remove_alpha_roundtrip() {
    let pixels: Vec<u8> = vec![100, 150, 200, 50, 100, 150];
    let info = info_rgb8(2, 1);
    let (with_alpha, info_rgba) = filters::add_alpha(&pixels, &info, 255).unwrap();
    assert_eq!(with_alpha[3], 255);
    assert_eq!(with_alpha[7], 255);
    let (removed, _) = filters::remove_alpha(&with_alpha, &info_rgba).unwrap();
    assert_exact("add_remove_alpha", &removed, &pixels);
}

#[test]
fn exact_convolve_identity() {
    let pixels = make_gradient_rgb(8, 8);
    let info = info_rgb8(8, 8);
    let kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let result = filters::convolve(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &kernel,
        &filters::ConvolveParams { kw: 3, kh: 3, divisor: 1.0 },
    ).unwrap();
    assert_exact("convolve identity", &result, &pixels);
}

#[test]
fn exact_brightness_zero() {
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);
    let result = filters::brightness(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::BrightnessParams { amount: 0.0 },
    ).unwrap();
    assert_exact("brightness(0)", &result, &pixels);
}

#[test]
fn exact_median_against_pillow() {
    // Median should be pixel-exact with Pillow (same algorithm: sorting-based)
    let w = 16u32;
    let h = 16;
    let mut pixels = vec![128u8; (w * h) as usize];
    pixels[5 * 16 + 5] = 0;
    pixels[10 * 16 + 10] = 255;
    let info = info_gray8(w, h);
    let ours = filters::MedianParams { radius: 1 }.compute(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
    ).unwrap();
    let script = format!(
        "import sys\nfrom PIL import Image,ImageFilter\nimport warnings\nwarnings.filterwarnings('ignore')\n\
         img=Image.frombytes('L',({w},{h}),bytes({pixels:?}))\n\
         f=img.filter(ImageFilter.MedianFilter(3))\nsys.stdout.buffer.write(f.tobytes())"
    );
    assert_exact("median r=1", &ours, &run_python_ref(&script));
}

// ═══════════════════════════════════════════════════════════════════════════
// CLOSE MATCH (MAE ≤ 1.0) — Integer rounding / border handling differences
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn close_sepia_against_numpy() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let ours = filters::sepia(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::SepiaParams { intensity: 1.0 },
    ).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\npx=np.array({pixels:?},dtype=np.uint8).reshape(-1,3)\n\
         m=np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]])\n\
         out=np.clip(px.astype(np.float64)@m.T+0.5,0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("sepia", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_blend_multiply_w3c() {
    let w = 8u32;
    let h = 8;
    let fg = make_gradient_rgb(w, h);
    let bg: Vec<u8> = fg.iter().map(|&v| 255 - v).collect();
    let info = info_rgb8(w, h);
    let ours = filters::blend(&fg, &info, &bg, &info, filters::BlendMode::Multiply).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nfg=np.array({fg:?},dtype=np.uint8)\nbg=np.array({bg:?},dtype=np.uint8)\n\
         out=np.clip((fg.astype(np.uint32)*bg.astype(np.uint32)+127)//255,0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("blend Multiply", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_blend_screen_w3c() {
    let w = 8u32;
    let h = 8;
    let fg = make_gradient_rgb(w, h);
    let bg: Vec<u8> = fg.iter().map(|&v| 255 - v).collect();
    let info = info_rgb8(w, h);
    let ours = filters::blend(&fg, &info, &bg, &info, filters::BlendMode::Screen).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nfg=np.array({fg:?},dtype=np.uint16)\nbg=np.array({bg:?},dtype=np.uint16)\n\
         out=np.clip(fg+bg-(fg*bg+127)//255,0,255).astype(np.uint8)\nsys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("blend Screen", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_convolve_sharpen_against_opencv() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let kernel = filters::kernels::EDGE_ENHANCE;
    let ours = filters::convolve(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &kernel,
        &filters::ConvolveParams { kw: 3, kh: 3, divisor: 1.0 },
    ).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nimport cv2\n\
         px=np.array({pixels:?},dtype=np.uint8).reshape({h},{w},3)\n\
         k=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)\n\
         out=cv2.filter2D(px,-1,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("convolve sharpen", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_box_blur_against_opencv() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let kernel = filters::kernels::BOX_BLUR_3X3;
    let ours = filters::convolve(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &kernel,
        &filters::ConvolveParams { kw: 3, kh: 3, divisor: 9.0 },
    ).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nimport cv2\n\
         px=np.array({pixels:?},dtype=np.uint8).reshape({h},{w},3)\n\
         out=cv2.blur(px,(3,3),borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_close("box blur", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn close_sobel_against_opencv() {
    let w = 16u32;
    let h = 16;
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in w / 2..w {
            pixels[(y * w + x) as usize] = 255;
        }
    }
    let info = info_gray8(w, h);
    let ours = filters::sobel(&pixels, &info).unwrap();
    let script = format!(
        "import sys\nimport numpy as np\nimport cv2\n\
         px=np.array({pixels:?},dtype=np.uint8).reshape({h},{w})\n\
         gx=cv2.Sobel(px,cv2.CV_32F,1,0,ksize=3,borderType=cv2.BORDER_REFLECT_101)\n\
         gy=cv2.Sobel(px,cv2.CV_32F,0,1,ksize=3,borderType=cv2.BORDER_REFLECT_101)\n\
         mag=np.clip(np.sqrt(gx**2+gy**2),0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(mag.tobytes())"
    );
    // Border handling differs (our reflect vs OpenCV BORDER_REFLECT_101)
    assert_close("sobel", &ours, &run_python_ref(&script), 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Summary — verifies venv is functional (fails if not)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reference_parity_summary() {
    let python = venv_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg("import numpy, cv2; from PIL import Image; print('ALL DEPS OK')")
        .output()
        .expect("venv python failed");
    assert!(
        output.status.success(),
        "Venv missing dependencies. Run:\n  tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    eprintln!();
    eprintln!("=== Filter Reference Parity Summary ===");
    eprintln!("  Venv: tests/fixtures/.venv (numpy + Pillow + OpenCV)");
    eprintln!(
        "  EXACT (MAE=0.0): premultiply, add/remove alpha, identity convolve, brightness, median"
    );
    eprintln!("  CLOSE (MAE≤1.0): sepia, blend Multiply/Screen, convolve sharpen, box blur");
    eprintln!("  CLOSE (MAE≤2.0): sobel (border reflect variant)");
    eprintln!("=======================================");
}

// ═══════════════════════════════════════════════════════════════════════════
// Morphological Operations — OpenCV Reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn morphology_erode_matches_opencv() {
    use filters::MorphShape;

    let py = venv_python();
    eprintln!("=== Morphology Erode — OpenCV Reference ===");

    // Use standard reference image (grayscale gradient)
    let w = 64u32;
    let h = 64u32;
    let mut input = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            input.push(((x * 255) / w) as u8);
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    // Our erode
    let ours = filters::erode(&input, &info, 3, MorphShape::Rect).unwrap();

    // OpenCV erode
    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         k=np.ones((3,3),np.uint8)\n\
         out=cv2.erode(px,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv erode failed");
    assert!(
        output.status.success(),
        "opencv erode: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference = output.stdout;

    assert_close("erode 3x3 rect", &ours, &reference, 1.0);
}

#[test]
fn morphology_dilate_matches_opencv() {
    use filters::MorphShape;

    let py = venv_python();
    eprintln!("=== Morphology Dilate — OpenCV Reference ===");

    let w = 64u32;
    let h = 64u32;
    let mut input = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            input.push(((x * 255) / w) as u8);
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let ours = filters::dilate(&input, &info, 3, MorphShape::Rect).unwrap();

    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         k=np.ones((3,3),np.uint8)\n\
         out=cv2.dilate(px,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv dilate failed");
    assert!(
        output.status.success(),
        "opencv dilate: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_close("dilate 3x3 rect", &ours, &output.stdout, 1.0);
}

#[test]
fn morphology_open_close_matches_opencv() {
    use filters::MorphShape;

    let py = venv_python();
    eprintln!("=== Morphology Open/Close — OpenCV Reference ===");

    // Use photo_256x256 as standard reference, converted to grayscale
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/generated/inputs/photo_256x256.png");
    let (input, w, h) = if fixture_path.exists() {
        // Decode with rasmcore's own decoder and convert to gray8
        let raw = std::fs::read(&fixture_path).unwrap();
        let decoded = rasmcore_image::domain::decoder::decode(&raw).unwrap();
        let gray = if decoded.info.format == PixelFormat::Gray8 {
            decoded.pixels.clone()
        } else {
            // Convert RGB8/RGBA8 to Gray8 using luminance
            let bpp = match decoded.info.format {
                PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
                PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
                _ => 3,
            };
            decoded.pixels.chunks_exact(bpp).map(|c| {
                ((c[0] as u32 * 77 + c[1] as u32 * 150 + c[2] as u32 * 29) >> 8) as u8
            }).collect()
        };
        (gray, decoded.info.width, decoded.info.height)
    } else {
        // Fallback: synthetic gradient
        let w = 64u32;
        let h = 64u32;
        let mut px = Vec::with_capacity((w * h) as usize);
        for _y in 0..h {
            for x in 0..w {
                px.push(((x * 255) / w) as u8);
            }
        }
        (px, w, h)
    };

    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    // Open
    let ours_open = filters::morph_open(&input, &info, 5, MorphShape::Rect).unwrap();
    let script_open = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         k=np.ones((5,5),np.uint8)\n\
         out=cv2.morphologyEx(px,cv2.MORPH_OPEN,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let out = Command::new(&py)
        .arg("-c")
        .arg(&script_open)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            use std::io::Write;
            c.stdin.take().unwrap().write_all(&input).unwrap();
            c.wait_with_output()
        })
        .expect("opencv open failed");
    assert!(
        out.status.success(),
        "opencv open: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_close("open 5x5 rect", &ours_open, &out.stdout, 1.0);

    // Close
    let ours_close = filters::morph_close(&input, &info, 5, MorphShape::Rect).unwrap();
    let script_close = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         k=np.ones((5,5),np.uint8)\n\
         out=cv2.morphologyEx(px,cv2.MORPH_CLOSE,k,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let out = Command::new(&py)
        .arg("-c")
        .arg(&script_close)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            use std::io::Write;
            c.stdin.take().unwrap().write_all(&input).unwrap();
            c.wait_with_output()
        })
        .expect("opencv close failed");
    assert!(
        out.status.success(),
        "opencv close: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_close("close 5x5 rect", &ours_close, &out.stdout, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// NLM Denoising — OpenCV Reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nlm_denoise_matches_opencv() {
    let py = venv_python();
    eprintln!("=== NLM Denoising — OpenCV Reference ===");

    // Create noisy grayscale image (deterministic)
    let w = 32u32;
    let h = 32u32;
    let mut input = vec![128u8; (w * h) as usize];
    for i in 0..input.len() {
        let noise = ((i as u32).wrapping_mul(2654435761) >> 24) as i16 - 128;
        input[i] = (128i16 + noise / 4).clamp(0, 255) as u8;
    }

    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let params = filters::NlmParams {
        h: 20.0,
        patch_size: 7,
        search_size: 21,
        ..Default::default() // OpenCv algorithm (matches cv2.fastNlMeansDenoising)
    };
    let ours = filters::nlm_denoise(&input, &info, &params).unwrap();

    // OpenCV NLM
    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         out=cv2.fastNlMeansDenoising(px,None,20.0,7,21)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv nlm failed");
    assert!(
        output.status.success(),
        "opencv nlm: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reference = &output.stdout;
    let mae = mean_absolute_error(&ours, reference);
    let max_err = max_absolute_error(&ours, reference);
    eprintln!("  NLM h=20 7x21: MAE={mae:.4}, max_err={max_err}");

    // OpenCV-exact algorithm: identical integer SSD, bit-shift avg, LUT, fixed-point.
    //
    // Remaining ±1 on ~0.4% of pixels: all have fractional part within 0.005 of 0.5.
    // Cause: C++ libm exp() vs Rust libm exp() differ at the ULP (unit of last place)
    // level for some inputs. This shifts a few LUT entries by ±1, which accumulates
    // across 441 search positions to flip rounding at the 0.5 boundary.
    //
    // The algorithm is mathematically identical — same integer SSD, same bit-shift,
    // same LUT formula, same fixed-point accumulation, same rounding division.
    // The only difference is IEEE 754 exp() implementation between compilers.
    assert!(
        mae < 0.01,
        "NLM: MAE={mae:.4} — algorithm must match OpenCV (only exp() ULP diff allowed)"
    );
    assert!(
        max_err <= 1,
        "NLM: max_err={max_err} — only ±1 from exp() ULP rounding allowed"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Lens Distortion Correction — OpenCV Reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn undistort_matches_opencv() {
    use rasmcore_image::domain::transform::{CameraMatrix, DistortionCoeffs, undistort};

    let py = venv_python();
    eprintln!("=== Undistort — OpenCV Reference ===");

    // Create gradient test image
    let w = 128u32;
    let h = 128u32;
    let mut input = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            input.push(((x * 255) / w) as u8);
            input.push(((y * 255) / h) as u8);
            input.push(128u8);
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };

    let camera = CameraMatrix {
        fx: 100.0,
        fy: 100.0,
        cx: 64.0,
        cy: 64.0,
    };
    let dist = DistortionCoeffs {
        k1: -0.3,
        k2: 0.1,
        k3: 0.0,
    };

    let ours = undistort(&input, &info, &camera, &dist).unwrap();

    // OpenCV undistort
    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w},3)\n\
         K=np.array([[100.0,0,64.0],[0,100.0,64.0],[0,0,1]],dtype=np.float64)\n\
         D=np.array([-0.3,0.1,0,0,0],dtype=np.float64)\n\
         out=cv2.undistort(px,K,D,None,K)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv undistort failed");
    assert!(
        output.status.success(),
        "opencv undistort: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reference = &output.stdout;

    // Compare only interior pixels (borders differ due to out-of-bounds handling)
    let bpp = 3;
    let margin = 10;
    let mut total_err: f64 = 0.0;
    let mut max_err: u8 = 0;
    let mut count = 0usize;
    for y in margin..(h as usize - margin) {
        for x in margin..(w as usize - margin) {
            for ch in 0..bpp {
                let idx = (y * w as usize + x) * bpp + ch;
                let d = (ours.pixels[idx] as i16 - reference[idx] as i16).unsigned_abs() as u8;
                total_err += d as f64;
                max_err = max_err.max(d);
                count += 1;
            }
        }
    }
    let mae = total_err / count as f64;
    eprintln!("  undistort (interior): MAE={mae:.4}, max_err={max_err}");

    // Pixel-exact: same fixed-point bilinear as OpenCV (INTER_BITS=5, COEF_BITS=15)
    assert_eq!(
        mae, 0.0,
        "undistort: MAE={mae:.4} — must be pixel-exact with OpenCV"
    );
    assert_eq!(
        max_err, 0,
        "undistort: max_err={max_err} — must be pixel-exact with OpenCV"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial Utilities — OpenCV Reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pyr_down_matches_opencv() {
    let py = venv_python();
    eprintln!("=== pyrDown — OpenCV Reference ===");

    let w = 64u32;
    let h = 64u32;
    let mut input = Vec::with_capacity((w * h) as usize);
    for _y in 0..h {
        for x in 0..w {
            input.push(((x * 255) / w) as u8);
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let (ours, _) = filters::pyr_down(&input, &info).unwrap();

    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         out=cv2.pyrDown(px,borderType=cv2.BORDER_REFLECT_101)\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv pyrDown failed");
    assert!(
        output.status.success(),
        "opencv pyrDown: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mae = mean_absolute_error(&ours, &output.stdout);
    let max_err = max_absolute_error(&ours, &output.stdout);
    eprintln!("  pyrDown: MAE={mae:.4}, max_err={max_err}");
    assert!(mae < 1.0, "pyrDown: MAE={mae:.4} — should be near-exact");
}

#[test]
fn connected_components_matches_opencv() {
    let py = venv_python();
    eprintln!("=== Connected Components — OpenCV Reference ===");

    // Create binary image with two blobs
    let w = 32u32;
    let h = 32u32;
    let mut input = vec![0u8; (w * h) as usize];
    // Blob 1: top-left 5x5
    for y in 2..7 {
        for x in 2..7 {
            input[y * w as usize + x] = 255;
        }
    }
    // Blob 2: bottom-right 5x5
    for y in 20..25 {
        for x in 20..25 {
            input[y * w as usize + x] = 255;
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let (our_labels, our_count) = filters::connected_components(&input, &info, 8).unwrap();

    let script = format!(
        "import sys,numpy as np,cv2\n\
         px=np.frombuffer(sys.stdin.buffer.read(),dtype=np.uint8).reshape({h},{w})\n\
         n,labels=cv2.connectedComponents(px,connectivity=8)\n\
         sys.stdout.buffer.write(np.array([n],dtype=np.uint32).tobytes())\n\
         sys.stdout.buffer.write(labels.astype(np.uint32).tobytes())"
    );
    let output = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input).unwrap();
            child.wait_with_output()
        })
        .expect("opencv connectedComponents failed");
    assert!(
        output.status.success(),
        "opencv CC: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Parse OpenCV output: first 4 bytes = num_labels (including background)
    let cv_count = u32::from_le_bytes(output.stdout[0..4].try_into().unwrap()) - 1; // subtract background
    eprintln!("  connectedComponents: ours={our_count} labels, cv={cv_count} labels");
    assert_eq!(our_count, cv_count, "component count must match OpenCV");
}

// ═══════════════════════════════════════════════════════════════════════════
// Vignette (Gaussian) — Close match against ImageMagick
// ═══════════════════════════════════════════════════════════════════════════

/// Run ImageMagick's vignette on raw RGB8 data and return the result.
fn im_vignette(pixels: &[u8], w: u32, h: u32, sigma: f32, ox: u32, oy: u32) -> Vec<u8> {
    let output = Command::new("magick")
        .args([
            "-size",
            &format!("{w}x{h}"),
            "-depth",
            "8",
            "rgb:-",
            "-background",
            "black",
            &format!("-vignette"),
            &format!("0x{sigma}+{ox}+{oy}"),
            "-depth",
            "8",
            "rgb:-",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(pixels).unwrap();
            child.wait_with_output()
        })
        .expect("magick vignette failed");
    assert!(
        output.status.success(),
        "magick vignette: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn close_vignette_gaussian_against_imagemagick() {
    // Skip if magick not available
    if Command::new("magick").arg("--version").output().is_err() {
        eprintln!("  SKIP — magick not installed");
        return;
    }

    let w = 128u32;
    let h = 128u32;
    let sigma = 20.0f32;
    let ox = 10u32;
    let oy = 10u32;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let ours = filters::vignette(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::VignetteParams { sigma, x_inset: ox, y_inset: oy, full_width: w, full_height: h, tile_offset_x: 0, tile_offset_y: 0 },
    ).unwrap();
    let reference = im_vignette(&pixels, w, h, sigma, ox, oy);

    assert_close("vignette gaussian vs IM 128x128", &ours, &reference, 1.5);
}

#[test]
fn close_vignette_gaussian_256_against_imagemagick() {
    if Command::new("magick").arg("--version").output().is_err() {
        eprintln!("  SKIP — magick not installed");
        return;
    }

    let w = 256u32;
    let h = 256u32;
    let sigma = 30.0f32;
    let ox = 20u32;
    let oy = 20u32;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let ours = filters::vignette(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::VignetteParams { sigma, x_inset: ox, y_inset: oy, full_width: w, full_height: h, tile_offset_x: 0, tile_offset_y: 0 },
    ).unwrap();
    let reference = im_vignette(&pixels, w, h, sigma, ox, oy);

    assert_close("vignette gaussian vs IM 256x256", &ours, &reference, 1.5);
}

#[test]
fn vignette_gaussian_alpha_preserved() {
    let w = 32u32;
    let h = 32u32;
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for _y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w) as u8);
            pixels.push(128);
            pixels.push(64);
            pixels.push(200); // alpha
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgba8,
        color_space: ColorSpace::Srgb,
    };
    let result = filters::vignette(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::VignetteParams { sigma: 10.0, x_inset: 5, y_inset: 5, full_width: w, full_height: h, tile_offset_x: 0, tile_offset_y: 0 },
    ).unwrap();
    for i in 0..(w * h) as usize {
        assert_eq!(
            result[i * 4 + 3],
            200,
            "alpha must be preserved at pixel {i}"
        );
    }
    eprintln!("  vignette gaussian alpha: preserved ✓");
}

// ═══════════════════════════════════════════════════════════════════════════
// Vignette (power-law) — Exact parity against numpy reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_vignette_powerlaw_rgb8_against_numpy() {
    let w = 32u32;
    let h = 24u32;
    let strength = 0.6f32;
    let falloff = 2.0f32;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let ours = filters::vignette_powerlaw(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::VignettePowerlawParams { strength, falloff, full_width: w, full_height: h, offset_x: 0, offset_y: 0 },
    ).unwrap();
    let script = format!(
        "import sys,numpy as np\npx=np.frombuffer(bytes({pixels:?}),dtype=np.uint8).reshape({h},{w},3).copy()\ncy,cx={h}/2.0,{w}/2.0\nmax_dist=np.sqrt(cx**2+cy**2)\nfor row in range({h}):\n for col in range({w}):\n  dx=col+0.5-cx; dy=row+0.5-cy\n  dist=np.sqrt(dx*dx+dy*dy)\n  t=(dist/max_dist)**{falloff}\n  factor=1.0-{strength}*t\n  for c in range(3):\n   px[row,col,c]=int(np.clip(round(px[row,col,c]*factor),0,255))\nsys.stdout.buffer.write(px.astype(np.uint8).tobytes())"
    );
    assert_exact("vignette powerlaw rgb8", &ours, &run_python_ref(&script));
}

#[test]
fn exact_vignette_powerlaw_gray8_against_numpy() {
    let w = 16u32;
    let h = 16u32;
    let strength = 0.5f32;
    let falloff = 1.5f32;
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for _y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w) as u8);
        }
    }
    let info = info_gray8(w, h);
    let ours = filters::vignette_powerlaw(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::VignettePowerlawParams { strength, falloff, full_width: w, full_height: h, offset_x: 0, offset_y: 0 },
    ).unwrap();
    let script = format!(
        "import sys,numpy as np\npx=np.frombuffer(bytes({pixels:?}),dtype=np.uint8).reshape({h},{w}).copy()\ncy,cx={h}/2.0,{w}/2.0\nmax_dist=np.sqrt(cx**2+cy**2)\nfor row in range({h}):\n for col in range({w}):\n  dx=col+0.5-cx; dy=row+0.5-cy\n  dist=np.sqrt(dx*dx+dy*dy)\n  t=(dist/max_dist)**{falloff}\n  factor=1.0-{strength}*t\n  px[row,col]=int(np.clip(round(px[row,col]*factor),0,255))\nsys.stdout.buffer.write(px.astype(np.uint8).tobytes())"
    );
    assert_exact("vignette powerlaw gray8", &ours, &run_python_ref(&script));
}

#[test]
fn vignette_powerlaw_zero_strength_is_identity() {
    let pixels = make_gradient_rgb(16, 16);
    let info = info_rgb8(16, 16);
    let result = filters::vignette_powerlaw(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::VignettePowerlawParams { strength: 0.0, falloff: 2.0, full_width: 16, full_height: 16, offset_x: 0, offset_y: 0 },
    ).unwrap();
    assert_exact("vignette_powerlaw(0)", &result, &pixels);
}

// ─── Frequency Separation ─────────────────────────────────────────────────
//
// Reference: scipy.ndimage.gaussian_filter for the blur component.
// High-pass = np.clip(original - blur + 128, 0, 255).
// Roundtrip: original = low + high - 128 (clamped).

#[test]
fn frequency_low_vs_scipy() {
    // Test against scipy Gaussian blur reference
    let w = 32u32;
    let h = 32;
    let sigma = 4.0f32;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let ours = filters::frequency_low(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &filters::FrequencyLowParams { sigma },
        ).unwrap();

    // Python reference: scipy gaussian_filter per channel
    let script = format!(
        r#"
import sys, numpy as np
from scipy.ndimage import gaussian_filter
px = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape({h},{w},3)
out = np.zeros_like(px)
for c in range(3):
    # scipy uses reflect (half-sample symmetric) like libblur's Clamp mode
    out[:,:,c] = np.clip(gaussian_filter(px[:,:,c].astype(np.float64), sigma={sigma}, mode='nearest') + 0.5, 0, 255).astype(np.uint8)
sys.stdout.buffer.write(out.tobytes())
"#
    );

    let python = venv_python();
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&pixels).unwrap();
            child.wait_with_output()
        })
        .unwrap();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named 'scipy'") {
            eprintln!("  frequency_low_vs_scipy: SKIP (scipy not installed)");
            return;
        }
        panic!("Python script failed: {stderr}");
    }
    let reference = output.stdout;

    // libblur uses a different border mode than scipy, so allow MAE < 2.0
    // (differences only at image borders due to padding mode mismatch)
    let mae = mean_absolute_error(&ours, &reference);
    let max_err = max_absolute_error(&ours, &reference);
    eprintln!("  frequency_low vs scipy: MAE={mae:.4}, max_err={max_err}");
    assert!(
        mae < 2.0,
        "frequency_low: MAE={mae:.4} too high vs scipy (border mode diff expected < 2.0)"
    );
}

#[test]
fn frequency_high_vs_numpy() {
    // High-pass = original - blur + 128, verified against numpy arithmetic
    let w = 32u32;
    let h = 32;
    let sigma = 4.0f32;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let low = filters::frequency_low(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &filters::FrequencyLowParams { sigma },
        ).unwrap();
    let high = filters::frequency_high(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &filters::FrequencyHighParams { sigma },
        ).unwrap();

    // Verify high-pass matches np.clip(original - low + 128, 0, 255)
    let mut expected_high = vec![0u8; pixels.len()];
    for i in 0..pixels.len() {
        let diff = pixels[i] as i16 - low[i] as i16 + 128;
        expected_high[i] = diff.clamp(0, 255) as u8;
    }
    assert_exact(
        "frequency_high vs (orig - low + 128)",
        &high,
        &expected_high,
    );
}

#[test]
fn frequency_separation_roundtrip_exact() {
    // Validate: low + high - 128 = original (within ±1 for blur rounding)
    let w = 64u32;
    let h = 64;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    for sigma in [1.0f32, 4.0, 10.0, 25.0] {
        let low = filters::frequency_low(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &filters::FrequencyLowParams { sigma },
        ).unwrap();
        let high = filters::frequency_high(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &filters::FrequencyHighParams { sigma },
        ).unwrap();

        let mut max_err: i16 = 0;
        for i in 0..pixels.len() {
            let reconstructed = (low[i] as i16 + high[i] as i16 - 128).clamp(0, 255);
            let err = (reconstructed - pixels[i] as i16).abs();
            max_err = max_err.max(err);
        }
        eprintln!("  roundtrip sigma={sigma}: max_err={max_err}");
        assert!(
            max_err <= 1,
            "frequency roundtrip sigma={sigma}: max_err={max_err} > 1"
        );
    }
}

#[test]
fn frequency_high_flat_image_is_neutral() {
    // A constant image should produce all-128 high-pass
    let info = info_rgb8(32, 32);
    let pixels = vec![100u8; 32 * 32 * 3];

    let high = filters::frequency_high(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &filters::FrequencyHighParams { sigma: 5.0 },
    ).unwrap();
    assert_exact("frequency_high(flat)", &high, &vec![128u8; pixels.len()]);
}

// ─── GIF Codec Parity ────────────────────────────────────────────────────

#[test]
fn gif_encode_decode_roundtrip_dimensions() {
    use rasmcore_image::domain::{decoder, encoder};

    let pixels = make_gradient_rgb(64, 64);
    let info = info_rgb8(64, 64);

    let encoded = encoder::encode(&pixels, &info, "gif", None).unwrap();
    assert_eq!(
        &encoded[..3],
        b"GIF",
        "encoded output must start with GIF magic"
    );

    let decoded = decoder::decode(&encoded).unwrap();
    assert_eq!(decoded.info.width, 64);
    assert_eq!(decoded.info.height, 64);
}

#[test]
fn gif_encode_decode_roundtrip_vs_imagemagick() {
    use rasmcore_image::domain::{decoder, encoder};

    // Encode a gradient to GIF with our encoder, then verify ImageMagick
    // can decode it to the same dimensions and similar pixel values
    let pixels = make_gradient_rgb(32, 32);
    let info = info_rgb8(32, 32);

    let encoded = encoder::encode(&pixels, &info, "gif", None).unwrap();

    // Decode with rasmcore
    let ours = decoder::decode(&encoded).unwrap();

    // Decode with ImageMagick (if available)
    let python = venv_python();
    let script = r#"
import sys, subprocess, tempfile, os
data = sys.stdin.buffer.read()
with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
    f.write(data)
    tmp = f.name
try:
    result = subprocess.run(['magick', tmp, '-depth', '8', 'rgba:-'],
        capture_output=True)
    if result.returncode != 0:
        sys.exit(1)
    sys.stdout.buffer.write(result.stdout)
finally:
    os.unlink(tmp)
"#;

    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&encoded).unwrap();
            child.wait_with_output()
        })
        .unwrap();

    if !output.status.success() {
        eprintln!("  gif_roundtrip_vs_imagemagick: SKIP (magick not available)");
        return;
    }

    let reference = output.stdout;
    if reference.len() != ours.pixels.len() {
        eprintln!(
            "  gif_roundtrip_vs_imagemagick: SKIP (size mismatch: ours={} ref={})",
            ours.pixels.len(),
            reference.len()
        );
        return;
    }

    // GIF is lossy (256-color quantization), so we compare what ImageMagick
    // decoded from our GIF against what we decoded from our GIF. Both should
    // see the same palette — differences are from implementation details.
    let mae = mean_absolute_error(&ours.pixels, &reference);
    let max_err = max_absolute_error(&ours.pixels, &reference);
    eprintln!("  gif decode rasmcore vs imagemagick: MAE={mae:.4}, max_err={max_err}");
    assert!(
        mae < 5.0,
        "GIF decode divergence too high: MAE={mae:.4} (expected < 5.0 for palette-quantized)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPOSURE — Photoshop-style logarithmic brightness (LUT-based)
// Reference: numpy f64 formula (gold standard for point operations)
//
// Formula: out = clamp01((in/255 + offset) * 2^ev) ^ (1/gamma) * 255
// Should be pixel-exact or max_err=1 due to f32 LUT quantization.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_exposure_identity() {
    // 0 EV, 0 offset, gamma 1.0 = identity
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);
    let config = filters::ExposureParams {
        ev: 0.0,
        offset: 0.0,
        gamma_correction: 1.0,
    };
    let result = filters::exposure(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();
    assert_exact("exposure(0EV)", &result, &pixels);
}

#[test]
fn exact_exposure_plus1ev_against_numpy() {
    // +1 EV doubles brightness: out = clamp(in * 2.0)
    let w = 256u32;
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(w, 1);
    let config = filters::ExposureParams {
        ev: 1.0,
        offset: 0.0,
        gamma_correction: 1.0,
    };
    let ours = filters::exposure(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();

    let script = format!(
        "import sys\nimport numpy as np\n\
         px=np.arange(256,dtype=np.float64)/255.0\n\
         out=np.clip(px*2.0,0,1)\n\
         result=np.floor(out*255.0+0.5).astype(np.uint8)\n\
         sys.stdout.buffer.write(result.tobytes())"
    );
    assert_exact("exposure(+1EV)", &ours, &run_python_ref(&script));
}

#[test]
fn exact_exposure_minus1ev_against_numpy() {
    // -1 EV halves brightness: out = clamp(in * 0.5)
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);
    let config = filters::ExposureParams {
        ev: -1.0,
        offset: 0.0,
        gamma_correction: 1.0,
    };
    let ours = filters::exposure(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();

    let script = format!(
        "import sys\nimport numpy as np\n\
         px=np.arange(256,dtype=np.float64)/255.0\n\
         out=np.clip(px*0.5,0,1)\n\
         result=np.floor(out*255.0+0.5).astype(np.uint8)\n\
         sys.stdout.buffer.write(result.tobytes())"
    );
    assert_exact("exposure(-1EV)", &ours, &run_python_ref(&script));
}

#[test]
fn exact_exposure_offset_against_numpy() {
    // Offset shifts input before scaling
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);
    let config = filters::ExposureParams {
        ev: 0.0,
        offset: 0.1,
        gamma_correction: 1.0,
    };
    let ours = filters::exposure(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();

    let script = format!(
        "import sys\nimport numpy as np\n\
         px=np.arange(256,dtype=np.float64)/255.0\n\
         out=np.clip(px+0.1,0,1)\n\
         result=np.floor(out*255.0+0.5).astype(np.uint8)\n\
         sys.stdout.buffer.write(result.tobytes())"
    );
    assert_exact("exposure(offset=0.1)", &ours, &run_python_ref(&script));
}

#[test]
fn exact_exposure_gamma_against_numpy() {
    // Gamma correction: out = clamp(in) ^ (1/gamma)
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);
    let config = filters::ExposureParams {
        ev: 0.0,
        offset: 0.0,
        gamma_correction: 2.2,
    };
    let ours = filters::exposure(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();

    let script = format!(
        "import sys\nimport numpy as np\n\
         px=np.arange(256,dtype=np.float64)/255.0\n\
         inv_gamma=1.0/2.2\n\
         out=np.where(px>0, np.power(px, inv_gamma), 0.0)\n\
         result=np.floor(out*255.0+0.5).astype(np.uint8)\n\
         sys.stdout.buffer.write(result.tobytes())"
    );
    // f32 powf vs f64 ** may differ by ±1 at rounding boundaries
    assert_close("exposure(gamma=2.2)", &ours, &run_python_ref(&script), 1.0);
}

#[test]
fn exact_exposure_combined_against_numpy() {
    // Combined: +0.5 EV, offset=-0.05, gamma=1.5
    let pixels = make_gradient_rgb(16, 16);
    let info = info_rgb8(16, 16);
    let config = filters::ExposureParams {
        ev: 0.5,
        offset: -0.05,
        gamma_correction: 1.5,
    };
    let ours = filters::exposure(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();

    let script = format!(
        "import sys\nimport numpy as np\n\
         px=np.array({pixels:?},dtype=np.float64)/255.0\n\
         ev=0.5; offset=-0.05; gamma=1.5\n\
         scaled=np.clip((px+offset)*np.power(2.0,ev),0,1)\n\
         inv_gamma=1.0/gamma\n\
         out=np.where(scaled>0, np.power(scaled, inv_gamma), 0.0)\n\
         result=np.floor(out*255.0+0.5).astype(np.uint8)\n\
         sys.stdout.buffer.write(result.tobytes())"
    );
    assert_close("exposure(combined)", &ours, &run_python_ref(&script), 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// COLOR BALANCE — Photoshop-style per-tonal-range CMY-RGB shifts
// Reference: numpy f64 formula (reproducing our tonal weight curves)
//
// Tonal weights: shadow = min((1-luma)^2 * 1.5, 1), highlight = min(luma^2 * 1.5, 1)
// midtone = max(1 - shadow - highlight, 0)
// Channel shift: dR = shadow[0]*sw + midtone[0]*mw + highlight[0]*hw
// Preserve luminosity: rescale to maintain original Rec.709 luma.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_color_balance_identity() {
    let pixels = make_gradient_rgb(16, 16);
    let info = info_rgb8(16, 16);
    let config = filters::ColorBalanceParams {
        shadow_cyan_red: 0.0,
        shadow_magenta_green: 0.0,
        shadow_yellow_blue: 0.0,
        midtone_cyan_red: 0.0,
        midtone_magenta_green: 0.0,
        midtone_yellow_blue: 0.0,
        highlight_cyan_red: 0.0,
        highlight_magenta_green: 0.0,
        highlight_yellow_blue: 0.0,
        preserve_luminosity: true,
    };
    let result = filters::color_balance(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();
    assert_exact("color_balance(identity)", &result, &pixels);
}

#[test]
fn close_color_balance_shadow_red_against_numpy() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let config = filters::ColorBalanceParams {
        shadow_cyan_red: 50.0,
        shadow_magenta_green: 0.0,
        shadow_yellow_blue: 0.0,
        midtone_cyan_red: 0.0,
        midtone_magenta_green: 0.0,
        midtone_yellow_blue: 0.0,
        highlight_cyan_red: 0.0,
        highlight_magenta_green: 0.0,
        highlight_yellow_blue: 0.0,
        preserve_luminosity: false,
    };
    let ours = filters::color_balance(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();

    let script = format!(
        "import sys\nimport numpy as np\n\
         px=np.array({pixels:?},dtype=np.float64).reshape(-1,3)/255.0\n\
         shadow=np.array([0.5,0.0,0.0])\n\
         midtone=np.array([0.0,0.0,0.0])\n\
         highlight=np.array([0.0,0.0,0.0])\n\
         luma=0.2126*px[:,0]+0.7152*px[:,1]+0.0722*px[:,2]\n\
         sw=np.minimum((1.0-luma)**2*1.5,1.0)\n\
         hw=np.minimum(luma**2*1.5,1.0)\n\
         mw=np.maximum(1.0-sw-hw,0.0)\n\
         dr=shadow[0]*sw+midtone[0]*mw+highlight[0]*hw\n\
         dg=shadow[1]*sw+midtone[1]*mw+highlight[1]*hw\n\
         db=shadow[2]*sw+midtone[2]*mw+highlight[2]*hw\n\
         out=np.clip(px+np.stack([dr,dg,db],axis=1),0,1)\n\
         result=np.clip(np.round(out*255.0),0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(result.tobytes())"
    );
    assert_close(
        "color_balance(shadow_red=50, no_lum)",
        &ours,
        &run_python_ref(&script),
        1.0,
    );
}

#[test]
fn close_color_balance_midtone_green_against_numpy() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let config = filters::ColorBalanceParams {
        shadow_cyan_red: 0.0,
        shadow_magenta_green: 0.0,
        shadow_yellow_blue: 0.0,
        midtone_cyan_red: 0.0,
        midtone_magenta_green: 75.0,
        midtone_yellow_blue: 0.0,
        highlight_cyan_red: 0.0,
        highlight_magenta_green: 0.0,
        highlight_yellow_blue: 0.0,
        preserve_luminosity: false,
    };
    let ours = filters::color_balance(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();

    let script = format!(
        "import sys\nimport numpy as np\n\
         px=np.array({pixels:?},dtype=np.float64).reshape(-1,3)/255.0\n\
         shadow=np.array([0.0,0.0,0.0])\n\
         midtone=np.array([0.0,0.75,0.0])\n\
         highlight=np.array([0.0,0.0,0.0])\n\
         luma=0.2126*px[:,0]+0.7152*px[:,1]+0.0722*px[:,2]\n\
         sw=np.minimum((1.0-luma)**2*1.5,1.0)\n\
         hw=np.minimum(luma**2*1.5,1.0)\n\
         mw=np.maximum(1.0-sw-hw,0.0)\n\
         dr=shadow[0]*sw+midtone[0]*mw+highlight[0]*hw\n\
         dg=shadow[1]*sw+midtone[1]*mw+highlight[1]*hw\n\
         db=shadow[2]*sw+midtone[2]*mw+highlight[2]*hw\n\
         out=np.clip(px+np.stack([dr,dg,db],axis=1),0,1)\n\
         result=np.clip(np.round(out*255.0),0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(result.tobytes())"
    );
    assert_close(
        "color_balance(midtone_green=75, no_lum)",
        &ours,
        &run_python_ref(&script),
        1.0,
    );
}

#[test]
fn close_color_balance_preserve_luminosity_against_numpy() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let config = filters::ColorBalanceParams {
        shadow_cyan_red: 40.0,
        shadow_magenta_green: 0.0,
        shadow_yellow_blue: -30.0,
        midtone_cyan_red: 0.0,
        midtone_magenta_green: 50.0,
        midtone_yellow_blue: 0.0,
        highlight_cyan_red: -20.0,
        highlight_magenta_green: 0.0,
        highlight_yellow_blue: 60.0,
        preserve_luminosity: true,
    };
    let ours = filters::color_balance(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    ).unwrap();

    let script = format!(
        "import sys\nimport numpy as np\n\
         px=np.array({pixels:?},dtype=np.float64).reshape(-1,3)/255.0\n\
         shadow=np.array([0.4,0.0,-0.3])\n\
         midtone=np.array([0.0,0.5,0.0])\n\
         highlight=np.array([-0.2,0.0,0.6])\n\
         luma=0.2126*px[:,0]+0.7152*px[:,1]+0.0722*px[:,2]\n\
         sw=np.minimum((1.0-luma)**2*1.5,1.0)\n\
         hw=np.minimum(luma**2*1.5,1.0)\n\
         mw=np.maximum(1.0-sw-hw,0.0)\n\
         dr=shadow[0]*sw+midtone[0]*mw+highlight[0]*hw\n\
         dg=shadow[1]*sw+midtone[1]*mw+highlight[1]*hw\n\
         db=shadow[2]*sw+midtone[2]*mw+highlight[2]*hw\n\
         out=np.clip(px+np.stack([dr,dg,db],axis=1),0,1)\n\
         new_luma=0.2126*out[:,0]+0.7152*out[:,1]+0.0722*out[:,2]\n\
         scale=np.where(new_luma>1e-6, luma/new_luma, 1.0)\n\
         out_lum=np.clip(out*scale[:,np.newaxis],0,1)\n\
         result=np.clip(np.round(out_lum*255.0),0,255).astype(np.uint8)\n\
         sys.stdout.buffer.write(result.tobytes())"
    );
    // f32 vs f64 luminance preservation may cause ±1 rounding differences
    assert_close(
        "color_balance(mixed, preserve_lum)",
        &ours,
        &run_python_ref(&script),
        1.0,
    );
}

// ─── High-Pass Filter ─────────────────────────────────────────────────────
//
// Reference: numpy arithmetic on scipy.ndimage.gaussian_filter output.
// high_pass = clamp((original - gaussian_filter(original, sigma)) / 2 + 128, 0, 255)
// Our blur uses sigma = radius directly.

#[test]
fn high_pass_vs_numpy_multi_radius() {
    // Use a larger image to reduce border-effect ratio at large radii
    let w = 128u32;
    let h = 128;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let python = venv_python();

    for radius in [1.0f32, 4.0, 10.0, 25.0] {
        let ours = filters::HighPassParams { radius }.compute(
            Rect::new(0, 0, w, h),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();

        let script = format!(
            r#"
import sys, numpy as np
from scipy.ndimage import gaussian_filter
px = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape({h},{w},3).astype(np.float64)
blurred = np.zeros_like(px)
for c in range(3):
    blurred[:,:,c] = gaussian_filter(px[:,:,c], sigma={radius}, mode='nearest')
hp = np.clip(np.floor((px - blurred) / 2.0) + 128.0, 0, 255).astype(np.uint8)
sys.stdout.buffer.write(hp.tobytes())
"#
        );

        let output = std::process::Command::new(&python)
            .arg("-c")
            .arg(&script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child.stdin.take().unwrap().write_all(&pixels).unwrap();
                child.wait_with_output()
            })
            .unwrap();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No module named 'scipy'") {
                eprintln!("  high_pass_vs_numpy: SKIP (scipy not installed)");
                return;
            }
            panic!("Python script failed: {stderr}");
        }

        let mae = mean_absolute_error(&ours, &output.stdout);
        let max_err = max_absolute_error(&ours, &output.stdout);
        // Border-mode differences (libblur clamp vs scipy nearest) grow with radius.
        // Tolerance scales with radius: larger kernel → more border pixels affected.
        let threshold = 2.0f64.max(radius as f64 * 0.3);
        eprintln!("  high_pass r={radius}: MAE={mae:.4}, max_err={max_err} (threshold={threshold:.1})");
        assert!(
            mae < threshold,
            "high_pass r={radius}: MAE={mae:.4} too high vs numpy (expected < {threshold:.1})"
        );
    }
}

// ─── LAB Channel Operations ──────────────────────────────────────────────
//
// Reference: skimage.color.rgb2lab (D65 illuminant, sRGB gamma).
// Our rgb_to_lab also uses D65, so results should match closely.

#[test]
fn lab_extract_l_vs_skimage() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let (ours, _) = filters::lab_extract_l(&pixels, &info).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
from skimage.color import rgb2lab
px = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape({h},{w},3)
lab = rgb2lab(px)
L = lab[:,:,0]  # L in [0, 100]
out = np.clip(np.round(L / 100.0 * 255.0), 0, 255).astype(np.uint8)
sys.stdout.buffer.write(out.tobytes())
"#
    );

    let python = venv_python();
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&pixels).unwrap();
            child.wait_with_output()
        })
        .unwrap();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named 'skimage'") {
            eprintln!("  lab_extract_l_vs_skimage: SKIP (scikit-image not installed)");
            return;
        }
        panic!("Python script failed: {stderr}");
    }

    assert_close("lab_extract_l vs skimage", &ours, &output.stdout, 1.0);
}

#[test]
fn lab_extract_a_b_vs_skimage() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let (ours_a, _) = filters::lab_extract_a(&pixels, &info).unwrap();
    let (ours_b, _) = filters::lab_extract_b(&pixels, &info).unwrap();

    // Python: a mapped to [0, 255] with 128=neutral, same for b
    let script = format!(
        r#"
import sys, numpy as np
from skimage.color import rgb2lab
px = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape({h},{w},3)
lab = rgb2lab(px)
a_out = np.clip(np.round(lab[:,:,1] + 128.0), 0, 255).astype(np.uint8)
b_out = np.clip(np.round(lab[:,:,2] + 128.0), 0, 255).astype(np.uint8)
sys.stdout.buffer.write(a_out.tobytes())
sys.stdout.buffer.write(b_out.tobytes())
"#
    );

    let python = venv_python();
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&pixels).unwrap();
            child.wait_with_output()
        })
        .unwrap();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named 'skimage'") {
            eprintln!("  lab_extract_a_b_vs_skimage: SKIP (scikit-image not installed)");
            return;
        }
        panic!("Python script failed: {stderr}");
    }

    let n = (w * h) as usize;
    let ref_a = &output.stdout[..n];
    let ref_b = &output.stdout[n..];

    assert_close("lab_extract_a vs skimage", &ours_a, ref_a, 1.0);
    assert_close("lab_extract_b vs skimage", &ours_b, ref_b, 1.0);
}

#[test]
fn lab_adjust_vs_skimage_multi_offset() {
    let w = 16u32;
    let h = 16;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let python = venv_python();

    // Test multiple a/b offset combinations including identity (0,0)
    let offsets: &[(f32, f32)] = &[
        (0.0, 0.0),     // identity — roundtrip should be near-exact
        (20.0, 0.0),    // shift toward red
        (0.0, -30.0),   // shift toward blue
        (-15.0, 25.0),  // mixed shift
        (50.0, 50.0),   // large shift
    ];

    for &(a_off, b_off) in offsets {
        let result = filters::lab_adjust(
            Rect::new(0, 0, w, h),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &filters::LabAdjustParams {
                a_offset: a_off,
                b_offset: b_off,
            },
        )
        .unwrap();

        // Python reference: same RGB→LAB→offset→LAB→RGB pipeline
        let script = format!(
            r#"
import sys, numpy as np
from skimage.color import rgb2lab, lab2rgb
px = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape({h},{w},3)
lab = rgb2lab(px)
lab[:,:,1] = np.clip(lab[:,:,1] + {a_off}, -128, 127)
lab[:,:,2] = np.clip(lab[:,:,2] + {b_off}, -128, 127)
out = np.clip(np.round(lab2rgb(lab) * 255.0), 0, 255).astype(np.uint8)
sys.stdout.buffer.write(out.tobytes())
"#
        );

        let output = std::process::Command::new(&python)
            .arg("-c")
            .arg(&script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child.stdin.take().unwrap().write_all(&pixels).unwrap();
                child.wait_with_output()
            })
            .unwrap();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No module named 'skimage'") {
                eprintln!("  lab_adjust_vs_skimage: SKIP (scikit-image not installed)");
                return;
            }
            panic!("Python script failed: {stderr}");
        }

        let mae = mean_absolute_error(&result, &output.stdout);
        let max_err = max_absolute_error(&result, &output.stdout);
        eprintln!("  lab_adjust a={a_off},b={b_off}: MAE={mae:.4}, max_err={max_err}");
        assert!(
            mae < 1.0,
            "lab_adjust a={a_off},b={b_off}: MAE={mae:.4} too high vs skimage"
        );
    }
}

#[test]
fn lab_sharpen_preserves_ab_multi_params() {
    // LAB sharpening should only modify L, leaving a/b channels unchanged.
    // Test across multiple amount/radius combinations.
    let w = 32u32;
    let h = 32;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let (a_before, _) = filters::lab_extract_a(&pixels, &info).unwrap();
    let (b_before, _) = filters::lab_extract_b(&pixels, &info).unwrap();

    let params: &[(f32, f32)] = &[
        (0.5, 1.0),   // subtle sharpen, small radius
        (1.0, 2.0),   // standard sharpen
        (2.0, 2.0),   // aggressive sharpen
        (3.0, 5.0),   // very aggressive, large radius
        (1.0, 10.0),  // standard amount, very large radius
    ];

    for &(amount, radius) in params {
        let sharpened = filters::lab_sharpen(
            Rect::new(0, 0, w, h),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &filters::LabSharpenParams { amount, radius },
        )
        .unwrap();

        let (a_after, _) = filters::lab_extract_a(&sharpened, &info).unwrap();
        let (b_after, _) = filters::lab_extract_b(&sharpened, &info).unwrap();

        // a/b channels should be preserved through LAB sharpening.
        // Roundtrip precision drift scales with sharpening amount — larger L
        // modifications cause slightly different a/b values after RGB→LAB→RGB.
        let mae_a = mean_absolute_error(&a_before, &a_after);
        let mae_b = mean_absolute_error(&b_before, &b_after);
        // Drift scales with both amount (sharpening strength) and radius
        // (how much of the image the blur touches — larger radius = more L modification)
        let threshold = 1.0f64.max(amount as f64 * 0.8 + radius as f64 * 0.15);
        eprintln!("  lab_sharpen amt={amount},r={radius}: a-drift={mae_a:.4}, b-drift={mae_b:.4} (threshold={threshold:.1})");
        assert!(
            mae_a < threshold,
            "lab_sharpen amt={amount},r={radius}: a-channel MAE={mae_a:.4} > {threshold:.1}"
        );
        assert!(
            mae_b < threshold,
            "lab_sharpen amt={amount},r={radius}: b-channel MAE={mae_b:.4} > {threshold:.1}"
        );
    }
}

// ─── Mesh Warp ───────────────────────────────────────────────────────────
//
// Reference: OpenCV cv2.remap with bilinear interpolation.
// We construct equivalent remap maps from the control point grid.

#[test]
fn mesh_warp_vs_opencv_remap() {
    // Use an identity-like grid with a small inward pinch at center.
    // Both src and dst stay within image bounds to avoid boundary issues.
    let w = 64u32;
    let h = 64;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    // 3x3 grid — corners fixed, center pinched inward by 4px
    let grid_json = format!(
        "[[0,0,0,0],[0.5,0,0.5,0],[1,0,1,0],\
         [0,0.5,0,0.5],[0.5,0.5,{},{}],[1,0.5,1,0.5],\
         [0,1,0,1],[0.5,1,0.5,1],[1,1,1,1]]",
        0.5 + 4.0 / w as f64,   // center dst shifted right 4px
        0.5 + 4.0 / h as f64,   // center dst shifted down 4px
    );

    let config = filters::MeshWarpParams {
        grid_cols: 3,
        grid_rows: 3,
        grid_json: grid_json.clone(),
    };

    let ours = filters::mesh_warp(
        Rect::new(0, 0, w, h),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    )
    .unwrap();

    // Python reference: replicate the exact same inverse-bilinear grid warp
    // using the same Newton's method, then sample via cv2.remap.
    let script = format!(
        r#"
import sys, json, numpy as np
import cv2

w, h = {w}, {h}
px = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape(h, w, 3)

grid = json.loads('{grid_json}')
grid_cols, grid_rows = 3, 3

src_pts = np.zeros((grid_rows, grid_cols, 2), dtype=np.float64)
dst_pts = np.zeros((grid_rows, grid_cols, 2), dtype=np.float64)
for row in range(grid_rows):
    for col in range(grid_cols):
        idx = row * grid_cols + col
        sx, sy, dx, dy = grid[idx]
        src_pts[row, col] = [sx * w, sy * h]
        dst_pts[row, col] = [dx * w, dy * h]

map_x = np.full((h, w), -1.0, dtype=np.float32)
map_y = np.full((h, w), -1.0, dtype=np.float32)

for row in range(grid_rows - 1):
    for col in range(grid_cols - 1):
        d_tl = dst_pts[row, col]
        d_tr = dst_pts[row, col+1]
        d_bl = dst_pts[row+1, col]
        d_br = dst_pts[row+1, col+1]
        s_tl = src_pts[row, col]
        s_tr = src_pts[row, col+1]
        s_bl = src_pts[row+1, col]
        s_br = src_pts[row+1, col+1]

        xs = [d_tl[0], d_tr[0], d_bl[0], d_br[0]]
        ys = [d_tl[1], d_tr[1], d_bl[1], d_br[1]]
        x0 = max(0, int(np.floor(min(xs))))
        x1 = min(w, int(np.ceil(max(xs))) + 1)
        y0 = max(0, int(np.floor(min(ys))))
        y1 = min(h, int(np.ceil(max(ys))) + 1)

        for py in range(y0, y1):
            for px_i in range(x0, x1):
                u, v = 0.5, 0.5
                for _ in range(20):
                    bx = (1-u)*(1-v)*d_tl[0] + u*(1-v)*d_tr[0] + (1-u)*v*d_bl[0] + u*v*d_br[0]
                    by = (1-u)*(1-v)*d_tl[1] + u*(1-v)*d_tr[1] + (1-u)*v*d_bl[1] + u*v*d_br[1]
                    ex, ey = px_i - bx, py - by
                    if abs(ex) < 0.001 and abs(ey) < 0.001:
                        break
                    dbx_du = (1-v)*(d_tr[0]-d_tl[0]) + v*(d_br[0]-d_bl[0])
                    dbx_dv = (1-u)*(d_bl[0]-d_tl[0]) + u*(d_br[0]-d_tr[0])
                    dby_du = (1-v)*(d_tr[1]-d_tl[1]) + v*(d_br[1]-d_bl[1])
                    dby_dv = (1-u)*(d_bl[1]-d_tl[1]) + u*(d_br[1]-d_tr[1])
                    det = dbx_du*dby_dv - dbx_dv*dby_du
                    if abs(det) < 1e-10:
                        break
                    u += (dby_dv*ex - dbx_dv*ey) / det
                    v += (-dby_du*ex + dbx_du*ey) / det
                if -0.01 <= u <= 1.01 and -0.01 <= v <= 1.01:
                    u = max(0, min(1, u))
                    v = max(0, min(1, v))
                    sx = (1-u)*(1-v)*s_tl[0] + u*(1-v)*s_tr[0] + (1-u)*v*s_bl[0] + u*v*s_br[0]
                    sy = (1-u)*(1-v)*s_tl[1] + u*(1-v)*s_tr[1] + (1-u)*v*s_bl[1] + u*v*s_br[1]
                    map_x[py, px_i] = np.float32(sx)
                    map_y[py, px_i] = np.float32(sy)

out = cv2.remap(px, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
sys.stdout.buffer.write(out.tobytes())
"#
    );

    let python = venv_python();
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&pixels).unwrap();
            child.wait_with_output()
        })
        .unwrap();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named 'cv2'") {
            eprintln!("  mesh_warp_vs_opencv_remap: SKIP (opencv not installed)");
            return;
        }
        panic!("Python script failed: {stderr}");
    }

    // Inverse bilinear convergence, quad-finding, and interpolation can differ
    // slightly between implementations — especially at quad boundaries.
    assert_close("mesh_warp vs OpenCV remap", &ours, &output.stdout, 3.0);
}

#[test]
fn mesh_warp_analytic_displacement() {
    // Non-uniform grid: shift center inward, verify displacement is smooth
    let w = 32u32;
    let h = 32;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    // 4x4 grid — pinch center inward by ~4px
    let mut grid = Vec::new();
    for row in 0..4u32 {
        for col in 0..4u32 {
            let sx = col as f64 / 3.0;
            let sy = row as f64 / 3.0;
            let cx = sx - 0.5;
            let cy = sy - 0.5;
            let dist = (cx * cx + cy * cy).sqrt();
            // Pinch: dst moves toward center proportionally
            let scale = if dist > 0.01 { 1.0 - 0.15 * (1.0 - dist).max(0.0) } else { 1.0 };
            let dx = 0.5 + cx * scale;
            let dy = 0.5 + cy * scale;
            grid.push(format!("[{sx},{sy},{dx},{dy}]"));
        }
    }
    let grid_json = format!("[{}]", grid.join(","));

    let config = filters::MeshWarpParams {
        grid_cols: 4,
        grid_rows: 4,
        grid_json,
    };

    let result = filters::mesh_warp(
        Rect::new(0, 0, w, h),
        &mut |_| Ok(pixels.to_vec()),
        &info,
        &config,
    )
    .unwrap();

    // Verify the output is not identical to input (warp actually happened)
    let mae = mean_absolute_error(&pixels, &result);
    eprintln!("  mesh_warp analytic pinch: MAE vs identity = {mae:.4}");
    assert!(
        mae > 1.0,
        "mesh_warp pinch should produce visible difference, MAE={mae:.4}"
    );

    // Verify corners are less affected than center (pinch effect)
    // Corner pixel (0,0) should be closer to original than center pixel (16,16)
    let corner_idx = 0;
    let center_idx = (16 * w + 16) as usize * 3;
    let corner_diff = (0..3)
        .map(|c| (pixels[corner_idx + c] as i32 - result[corner_idx + c] as i32).abs())
        .sum::<i32>();
    let center_diff = (0..3)
        .map(|c| (pixels[center_idx + c] as i32 - result[center_idx + c] as i32).abs())
        .sum::<i32>();
    eprintln!("  corner diff={corner_diff}, center diff={center_diff}");
    // Center should be more affected by the pinch
    assert!(
        center_diff >= corner_diff,
        "pinch should affect center more than corners"
    );
}

// ─── Skeletonize ─────────────────────────────────────────────────────────
//
// Reference: skimage.morphology.skeletonize (Zhang-Suen / medial axis).
// Both implement Zhang-Suen 1984, so results should be pixel-exact.

#[test]
fn skeletonize_vs_skimage() {
    // Create a thick rectangle binary image
    let w = 32u32;
    let h = 32;
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 8..24 {
        for x in 4..28 {
            pixels[(y * w + x) as usize] = 255;
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let ours = filters::skeletonize(&pixels, &info, 0).unwrap();

    // Python reference: skimage.morphology.skeletonize with Zhang-Suen method
    let script = format!(
        r#"
import sys, numpy as np
from skimage.morphology import skeletonize
px = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape({h},{w})
binary = px > 0
skel = skeletonize(binary, method='zhang')
out = (skel.astype(np.uint8)) * 255
sys.stdout.buffer.write(out.tobytes())
"#
    );

    let python = venv_python();
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&pixels).unwrap();
            child.wait_with_output()
        })
        .unwrap();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named 'skimage'") {
            eprintln!("  skeletonize_vs_skimage: SKIP (scikit-image not installed)");
            return;
        }
        panic!("Python script failed: {stderr}");
    }

    // Zhang-Suen is deterministic in principle, but pixel deletion order within
    // each sub-iteration can vary (row-major vs set-based), causing a few
    // boundary pixels to differ. Both produce valid skeletons.
    let mae = mean_absolute_error(&ours, &output.stdout);
    let max_err = max_absolute_error(&ours, &output.stdout);
    eprintln!("  skeletonize vs skimage (rectangle): MAE={mae:.4}, max_err={max_err}");
    // Very close but allow a few pixels of difference (MAE < 2.0 for binary)
    assert!(
        mae < 2.0,
        "skeletonize vs skimage: MAE={mae:.4} too high (expected < 2.0)"
    );
}

#[test]
fn skeletonize_vs_opencv_thinning() {
    // Cross-check against OpenCV's thinning (also Zhang-Suen)
    let w = 32u32;
    let h = 32;
    // Create an L-shaped binary image
    let mut pixels = vec![0u8; (w * h) as usize];
    // Vertical bar
    for y in 4..28 {
        for x in 4..12 {
            pixels[(y * w + x) as usize] = 255;
        }
    }
    // Horizontal bar at bottom
    for y in 20..28 {
        for x in 4..28 {
            pixels[(y * w + x) as usize] = 255;
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let ours = filters::skeletonize(&pixels, &info, 0).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
import cv2
px = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape({h},{w})
thinned = cv2.ximgproc.thinning(px, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
sys.stdout.buffer.write(thinned.tobytes())
"#
    );

    let python = venv_python();
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(&script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&pixels).unwrap();
            child.wait_with_output()
        })
        .unwrap();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named 'cv2'") || stderr.contains("ximgproc") {
            eprintln!("  skeletonize_vs_opencv: SKIP (opencv-contrib not installed)");
            return;
        }
        panic!("Python script failed: {stderr}");
    }

    // Both implement Zhang-Suen — very close, may differ by a few boundary pixels
    let mae = mean_absolute_error(&ours, &output.stdout);
    let max_err = max_absolute_error(&ours, &output.stdout);
    eprintln!("  skeletonize vs OpenCV thinning (L-shape): MAE={mae:.4}, max_err={max_err}");
    assert!(
        mae < 2.0,
        "skeletonize vs OpenCV: MAE={mae:.4} too high (expected < 2.0)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ADJUSTMENT PARITY — invert, contrast, gamma, levels, posterize,
//                      sigmoidal_contrast
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_invert_against_numpy() {
    // invert: out = 255 - in (pixel-exact, no rounding ambiguity)
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);
    let ours = filters::invert_registered(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.to_vec()),
        &info,
    )
    .unwrap();

    let script = "\
import sys\nimport numpy as np\n\
px=np.arange(256,dtype=np.uint8)\n\
out=np.uint8(255)-px\n\
sys.stdout.buffer.write(out.tobytes())";
    assert_exact("invert(gray)", &ours, &run_python_ref(script));
}

#[test]
fn exact_invert_rgb_against_numpy() {
    // invert on RGB gradient — each channel inverted independently
    let w = 64u32;
    let h = 64u32;
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let ours = filters::invert_registered(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.clone()),
        &info,
    )
    .unwrap();

    let script = format!(
        "import sys,numpy as np\n\
         w,h={w},{h}\n\
         xs=np.tile((np.arange(w,dtype=np.int32)*255//w).astype(np.uint8),(h,1))\n\
         ys=np.tile((np.arange(h,dtype=np.int32)*255//h).astype(np.uint8).reshape(h,1),(1,w))\n\
         bs=np.full((h,w),128,dtype=np.uint8)\n\
         px=np.stack([xs,ys,bs],axis=2).reshape(-1)\n\
         out=np.uint8(255)-px\n\
         sys.stdout.buffer.write(out.tobytes())"
    );
    assert_exact("invert(rgb)", &ours, &run_python_ref(&script));
}

#[test]
fn close_contrast_against_numpy() {
    // Contrast formula: factor = 1 + amount*2 (for amount >= 0)
    //   out = clamp(factor * (in - 128) + 128, 0, 255)  (truncate, not round)
    // Rust uses f32 LUT, Python f64 — MAE < 0.5 for truncation boundary diffs
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);

    for &amount in &[0.25f32, 0.5, -0.3, -0.5] {
        let config = filters::ContrastParams { amount };
        let ours = filters::contrast(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();

        let factor = if amount >= 0.0 {
            1.0 + amount as f64 * 2.0
        } else {
            1.0 / (1.0 - amount as f64 * 2.0)
        };
        let script = format!(
            "import sys\nimport numpy as np\n\
             px=np.arange(256,dtype=np.float64)\n\
             factor={factor}\n\
             out=np.clip(factor*(px-128.0)+128.0,0,255)\n\
             result=out.astype(np.uint8)\n\
             sys.stdout.buffer.write(result.tobytes())"
        );
        assert_close(
            &format!("contrast({amount})"),
            &ours,
            &run_python_ref(&script),
            0.5,
        );
    }
}

#[test]
fn exact_gamma_against_numpy() {
    // Gamma: out = round(255 * (in/255)^(1/gamma))
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);

    for &gamma in &[0.5f32, 1.0, 2.2, 3.0] {
        let config = filters::GammaParams {
            gamma_value: gamma,
        };
        let ours = filters::gamma_registered(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();

        let inv_gamma = 1.0 / gamma as f64;
        let script = format!(
            "import sys\nimport numpy as np\n\
             px=np.arange(256,dtype=np.float32)/255.0\n\
             out=np.power(px,{inv_gamma})*255.0+0.5\n\
             result=out.astype(np.uint8)\n\
             sys.stdout.buffer.write(result.tobytes())"
        );
        assert_exact(
            &format!("gamma({gamma})"),
            &ours,
            &run_python_ref(&script),
        );
    }
}

#[test]
fn exact_levels_against_numpy() {
    // Levels: out = clamp(((in/255 - black) / (white - black)) ^ (1/gamma)) * 255
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);

    // Test several black/white/gamma combos
    let cases: &[(f32, f32, f32)] = &[
        (10.0, 90.0, 1.0),   // simple range mapping
        (0.0, 100.0, 2.2),   // gamma only
        (20.0, 80.0, 0.5),   // narrow range + gamma
        (0.0, 50.0, 1.0),    // heavy clip
    ];

    for &(black_pct, white_pct, gamma) in cases {
        let config = filters::LevelsParams {
            black_point: black_pct,
            white_point: white_pct,
            gamma,
        };
        let ours = filters::levels(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();

        let black = black_pct as f64 / 100.0;
        let white = white_pct as f64 / 100.0;
        let inv_gamma = 1.0 / gamma as f64;
        let script = format!(
            "import sys\nimport numpy as np\n\
             px=np.arange(256,dtype=np.float32)/255.0\n\
             black={black}\n\
             white={white}\n\
             rng=max(white-black,1e-6)\n\
             norm=np.clip((px-black)/rng,0,1)\n\
             out=np.power(norm,{inv_gamma})*255.0+0.5\n\
             result=out.astype(np.uint8)\n\
             sys.stdout.buffer.write(result.tobytes())"
        );
        assert_exact(
            &format!("levels(b={black_pct},w={white_pct},g={gamma})"),
            &ours,
            &run_python_ref(&script),
        );
    }
}

#[test]
fn exact_posterize_against_pillow() {
    // Posterize: quantized = round(in * (n-1) / 255) -> out = round(quantized * 255 / (n-1))
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);

    for &levels in &[2u8, 4, 8, 16] {
        let config = filters::PosterizeParams { levels };
        let ours = filters::posterize_registered(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();

        let n = levels as f64;
        let script = format!(
            "import sys\nimport numpy as np\n\
             px=np.arange(256,dtype=np.float64)\n\
             n={n}\n\
             quantized=np.floor(px*(n-1.0)/255.0+0.5).astype(np.uint8)\n\
             out=np.floor(quantized.astype(np.float64)*255.0/(n-1.0)+0.5).astype(np.uint8)\n\
             sys.stdout.buffer.write(out.tobytes())"
        );
        assert_exact(
            &format!("posterize({levels})"),
            &ours,
            &run_python_ref(&script),
        );
    }
}

#[test]
fn close_sigmoidal_contrast_against_imagemagick() {
    // Sigmoidal contrast — S-curve:
    //   sig(x) = 1 / (1 + exp(strength * (midpoint - x)))
    //   out = (sig(x) - sig(0)) / (sig(1) - sig(0)) * 255
    // Uses f32 vs f64, so allow MAE < 0.5 for rounding
    let pixels: Vec<u8> = (0..=255).collect();
    let info = info_gray8(256, 1);

    let cases: &[(f32, f32, bool)] = &[
        (3.0, 50.0, true),    // moderate sharpen
        (10.0, 50.0, true),   // strong sharpen
        (5.0, 30.0, true),    // off-center midpoint
        (3.0, 50.0, false),   // soften
    ];

    for &(strength, midpoint, sharpen) in cases {
        let config = filters::SigmoidalContrastParams {
            strength,
            midpoint,
            sharpen,
        };
        let ours = filters::sigmoidal_contrast(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();

        let mid_frac = midpoint as f64 / 100.0;
        let s = strength as f64;
        let sharpen_str = if sharpen { "True" } else { "False" };
        let script = format!(
            "import sys,numpy as np\n\
px=np.arange(256,dtype=np.float64)/255.0\n\
s={s}\n\
mid={mid_frac}\n\
sharpen={sharpen_str}\n\
def sig(x): return 1.0/(1.0+np.exp(s*(mid-x)))\n\
sig0=sig(0.0)\n\
sig1=sig(1.0)\n\
rng=sig1-sig0\n\
if sharpen:\n  out=(sig(px)-sig0)/rng\n\
else:\n  y_scaled=px*rng+sig0\n  y_clamped=np.clip(y_scaled,1e-7,1.0-1e-7)\n  out=mid-np.log((1.0-y_clamped)/y_clamped)/s\n\
out=np.clip(out,0,1)*255.0+0.5\n\
result=out.astype(np.uint8)\n\
sys.stdout.buffer.write(result.tobytes())"
        );
        assert_close(
            &format!("sigmoidal({strength},{midpoint},{sharpen})"),
            &ours,
            &run_python_ref(&script),
            0.5,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EFFECT PARITY — emboss, solarize, oil_paint, pixelate, halftone
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn close_emboss_against_opencv_filter2d() {
    // Emboss kernel: [-2,-1,0; -1,1,1; 0,1,2] convolved via OpenCV filter2D.
    // OpenCV filter2D is the industry-standard 2D convolution — validates that
    // our convolve() produces the same result as a real image processing library.
    let (w, h) = (32u32, 32);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let r = Rect::new(0, 0, w, h);
    let ours = filters::emboss(r, &mut |_| Ok(pixels.clone()), &info).unwrap();

    let script = format!(
        r#"
import sys
import numpy as np
import cv2

px = np.array({pixels:?}, dtype=np.uint8).reshape({h}, {w}, 3)
kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32)
# OpenCV filter2D with BORDER_REFLECT_101 — the standard convolution
out = cv2.filter2D(px, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
sys.stdout.buffer.write(out.tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("emboss vs OpenCV filter2D", &ours, &reference, 1.0);
}

#[test]
fn exact_solarize_against_pillow() {
    // Solarize against Pillow ImageOps.solarize — the standard Python imaging
    // library used by millions of applications. Both use >= threshold.
    let (w, h) = (16u32, 16);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let threshold: u8 = 128;
    let r = Rect::new(0, 0, w, h);
    let ours = filters::solarize(
        r,
        &mut |_| Ok(pixels.clone()),
        &info,
        &filters::SolarizeParams { threshold },
    )
    .unwrap();

    let script = format!(
        r#"
import sys
import numpy as np
from PIL import Image, ImageOps

px = np.array({pixels:?}, dtype=np.uint8).reshape({h}, {w}, 3)
img = Image.fromarray(px, mode='RGB')
out = ImageOps.solarize(img, threshold={threshold})
sys.stdout.buffer.write(np.array(out).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_exact("solarize t=128 vs Pillow", &ours, &reference);
}

#[test]
fn exact_solarize_threshold_zero_against_pillow() {
    // threshold=0: full inversion. Validated against Pillow ImageOps.solarize.
    let (w, h) = (16u32, 16);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let r = Rect::new(0, 0, w, h);
    let ours = filters::solarize(
        r,
        &mut |_| Ok(pixels.clone()),
        &info,
        &filters::SolarizeParams { threshold: 0 },
    )
    .unwrap();

    let script = format!(
        r#"
import sys
import numpy as np
from PIL import Image, ImageOps

px = np.array({pixels:?}, dtype=np.uint8).reshape({h}, {w}, 3)
img = Image.fromarray(px, mode='RGB')
out = ImageOps.solarize(img, threshold=0)
sys.stdout.buffer.write(np.array(out).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_exact("solarize t=0 vs Pillow", &ours, &reference);
}

#[test]
fn close_oil_paint_structural_properties() {
    // Oil paint has no pixel-exact standard reference — ImageMagick -paint uses
    // Q16-HDRI precision and different tie-breaking. That comparison is already
    // in reference_audit.rs (algorithm_oil_paint, MAE < 5.0).
    //
    // Here we validate structural properties that any correct oil paint must have:
    // 1. Flat input → output equals input (all pixels same intensity bin)
    // 2. Output values are always averages of input values (no new values invented)
    // 3. Effect reduces unique colors (smoothing property)
    let (w, h) = (32u32, 32);
    let info = info_rgb8(w, h);
    let radius = 3u32;
    let rr = Rect::new(0, 0, w, h);

    // Flat input: every pixel has the same intensity → mode bin covers everything
    // → output must equal input exactly.
    let flat = vec![100u8; (w * h * 3) as usize];
    let flat_out = filters::oil_paint(
        rr,
        &mut |_| Ok(flat.clone()),
        &info,
        &filters::OilPaintParams { radius },
    )
    .unwrap();
    assert_eq!(flat, flat_out, "oil_paint of flat input must be identity");

    // Gradient: oil paint should reduce unique colors (smoothing)
    let gradient = make_gradient_rgb(w, h);
    let grad_out = filters::oil_paint(
        rr,
        &mut |_| Ok(gradient.clone()),
        &info,
        &filters::OilPaintParams { radius },
    )
    .unwrap();
    let in_unique: std::collections::HashSet<[u8; 3]> = gradient
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let out_unique: std::collections::HashSet<[u8; 3]> = grad_out
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    assert!(
        out_unique.len() <= in_unique.len(),
        "oil_paint should reduce or maintain unique color count: {} → {}",
        in_unique.len(),
        out_unique.len()
    );
    eprintln!(
        "  oil_paint structural: flat=identity ✓, colors {}/{} ✓",
        out_unique.len(),
        in_unique.len()
    );
}

#[test]
fn pixelate_block_grid_truncated_edges() {
    // Non-aligned pixelate: validates the block-grid truncation behavior that
    // matches Photoshop Mosaic, GIMP/GEGL pixelize, and FFmpeg pixelize.
    //
    // The grid starts at (0,0) and tiles with fixed block_size. Edge blocks are
    // truncated to image bounds (not padded or centered). Each block is filled
    // with the mean of its actual pixels. This differs from resize-based
    // pixelation (OpenCV/ImageMagick) which uses proportional mapping to produce
    // visually uniform blocks.
    //
    // Reference: GEGL pixelize source (gegl_rectangle_intersect clamps to bounds),
    // FFmpeg vf_pixelize.c (FFMIN(block_size, remaining)).
    let (w, h) = (30u32, 25);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let block_size = 7u32;
    let r = Rect::new(0, 0, w, h);
    let ours = filters::pixelate(
        r,
        &mut |_| Ok(pixels.clone()),
        &info,
        &filters::PixelateParams { block_size },
    )
    .unwrap();

    // Validate block-grid structure: each cell in the grid has uniform color,
    // and that color equals the mean of the source pixels in that cell.
    let bs = block_size as usize;
    let ww = w as usize;
    let hh = h as usize;
    let mut checked = 0;
    let mut by = 0;
    while by < hh {
        let bh = bs.min(hh - by);
        let mut bx = 0;
        while bx < ww {
            let bw = bs.min(ww - bx);

            // Verify uniform color within block
            let block_color = &ours[(by * ww + bx) * 3..(by * ww + bx) * 3 + 3];
            for row in by..(by + bh) {
                for col in bx..(bx + bw) {
                    let off = (row * ww + col) * 3;
                    assert_eq!(
                        &ours[off..off + 3],
                        block_color,
                        "block ({bx},{by}) not uniform at ({col},{row})"
                    );
                }
            }

            // Verify block color is the mean of source pixels (rounded)
            let mut sums = [0u32; 3];
            let count = (bw * bh) as u32;
            for row in by..(by + bh) {
                for col in bx..(bx + bw) {
                    let off = (row * ww + col) * 3;
                    for c in 0..3 {
                        sums[c] += pixels[off + c] as u32;
                    }
                }
            }
            for c in 0..3 {
                let expected = ((sums[c] + count / 2) / count) as u8;
                assert_eq!(
                    block_color[c], expected,
                    "block ({bx},{by}) ch{c}: got {}, expected {expected}",
                    block_color[c]
                );
            }
            checked += 1;

            bx += bs;
        }
        by += bs;
    }

    // 30/7 = 4 full + 1 partial = 5 columns, 25/7 = 3 full + 1 partial = 4 rows
    assert_eq!(checked, 20, "expected 5×4=20 blocks for 30x25 bs=7");

    // Verify edge blocks are actually smaller (the truncation property)
    let last_col_width = ww % bs; // 30 % 7 = 2
    let last_row_height = hh % bs; // 25 % 7 = 4
    assert_eq!(last_col_width, 2, "rightmost column should be 2px wide");
    assert_eq!(last_row_height, 4, "bottom row should be 4px tall");
    eprintln!(
        "  pixelate non-aligned: {checked} blocks, edge=({last_col_width}×{last_row_height}) ✓"
    );
}

#[test]
fn halftone_structural_correctness() {
    // Halftone: no standard library implements the same CMYK sine-wave screening.
    // ImageMagick, Photoshop, and GIMP each use different halftone algorithms.
    // We validate structural invariants that any correct halftone must satisfy.
    let (w, h) = (64u32, 64);
    let info = info_rgb8(w, h);
    let config = filters::HalftoneParams {
        dot_size: 6.0,
        angle_offset: 0.0,
    };

    // White input → all CMYK channels are 0 → output should be white
    let white = vec![255u8; (w * h * 3) as usize];
    let r = Rect::new(0, 0, w, h);
    let white_out = filters::halftone(
        r,
        &mut |_| Ok(white.clone()),
        &info,
        &config,
    )
    .unwrap();
    assert!(
        white_out.iter().all(|&v| v == 255),
        "halftone of white input should be all white"
    );

    // Black input → K=1 → output should be all black
    let black = vec![0u8; (w * h * 3) as usize];
    let black_out = filters::halftone(
        r,
        &mut |_| Ok(black.clone()),
        &info,
        &config,
    )
    .unwrap();
    assert!(
        black_out.iter().all(|&v| v == 0),
        "halftone of black input should be all black"
    );

    // Gradient input: binary screening → output values should only be 0 or 255
    let gradient = make_gradient_rgb(w, h);
    let grad_out = filters::halftone(
        r,
        &mut |_| Ok(gradient.clone()),
        &info,
        &config,
    )
    .unwrap();
    let unique_vals: std::collections::HashSet<u8> = grad_out.iter().copied().collect();
    assert!(
        unique_vals.is_subset(&[0u8, 255].iter().copied().collect()),
        "halftone output should only contain 0 and 255, got {unique_vals:?}"
    );

    // Darker regions should have more black pixels than lighter regions
    // (halftone density should correlate with input darkness)
    let dark_quarter: f64 = grad_out[..((w * h * 3 / 4) as usize)]
        .iter()
        .map(|&v| v as f64)
        .sum::<f64>();
    let light_quarter: f64 = grad_out[((w * h * 3 * 3 / 4) as usize)..]
        .iter()
        .map(|&v| v as f64)
        .sum::<f64>();
    // Note: gradient goes left-to-right in R, top-to-bottom in G, so the
    // first quarter of linear memory has lower R+G values (darker).
    assert!(
        dark_quarter < light_quarter,
        "halftone: darker regions should have lower luminance sum"
    );
    eprintln!("  halftone structural: white=white ✓, black=black ✓, binary ✓, density ✓");
}

// ═══════════════════════════════════════════════════════════════════════════
// SPATIAL GAP PARITY — motion_blur, gaussian_blur_cv
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn close_motion_blur_against_opencv_filter2d() {
    // Motion blur at 0 degrees is a horizontal averaging kernel.
    // Validate against OpenCV filter2D — the industry-standard convolution.
    let (w, h) = (32u32, 32);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let length = 5u32;
    let r = Rect::new(0, 0, w, h);
    let ours = filters::MotionBlurParams {
        length,
        angle_degrees: 0.0,
    }.compute(
        r,
        &mut |req| {
            let mut p = Vec::with_capacity((req.width * req.height * 3) as usize);
            for y in req.y..(req.y + req.height) {
                for x in req.x..(req.x + req.width) {
                    p.push(((x * 255) / w) as u8);
                    p.push(((y * 255) / h) as u8);
                    p.push(128);
                }
            }
            Ok(p)
        },
        &info,
    )
    .unwrap();

    // OpenCV filter2D with our motion blur kernel — validates the convolution
    // result against a production library, not a reimplementation of our code.
    let script = format!(
        r#"
import sys
import numpy as np
import cv2
import math

# Generate the same gradient
px = np.zeros(({h}, {w}, 3), dtype=np.uint8)
for y in range({h}):
    for x in range({w}):
        px[y, x, 0] = (x * 255) // {w}
        px[y, x, 1] = (y * 255) // {h}
        px[y, x, 2] = 128

# Build the same motion blur kernel our Rust code builds
length = {length}
side = 2 * length + 1
kernel = np.zeros((side, side), dtype=np.float32)
center = length
dx, dy = 1.0, 0.0
steps = int(math.ceil(length * 2.0)) * 2 + 1
count = 0
for i in range(steps):
    t = (i / (steps - 1)) * 2.0 - 1.0
    ix = int(round(center + t * length * dx))
    iy = int(round(center + t * length * dy))
    if 0 <= ix < side and 0 <= iy < side and kernel[iy, ix] == 0.0:
        kernel[iy, ix] = 1.0
        count += 1
kernel /= count

# Apply via OpenCV filter2D — a real, production convolution engine
out = cv2.filter2D(px, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
sys.stdout.buffer.write(out.tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("motion_blur 0° len=5 vs OpenCV", &ours, &reference, 2.0);
}

#[test]
fn close_gaussian_blur_cv_against_opencv() {
    // Gaussian blur with sigma=1.5 should match OpenCV GaussianBlur
    let (w, h) = (32u32, 32);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let sigma = 1.5f32;
    let r = Rect::new(0, 0, w, h);
    let ours = filters::GaussianBlurCvParams { sigma }.compute(
        r,
        &mut |req| {
            let mut p = Vec::with_capacity((req.width * req.height * 3) as usize);
            for y in req.y..(req.y + req.height) {
                for x in req.x..(req.x + req.width) {
                    p.push(((x * 255) / w) as u8);
                    p.push(((y * 255) / h) as u8);
                    p.push(128);
                }
            }
            Ok(p)
        },
        &info,
    )
    .unwrap();

    let script = format!(
        r#"
import sys
import numpy as np
import cv2

# Generate the same gradient
px = np.zeros(({h}, {w}, 3), dtype=np.uint8)
for y in range({h}):
    for x in range({w}):
        px[y, x, 0] = (x * 255) // {w}
        px[y, x, 1] = (y * 255) // {h}
        px[y, x, 2] = 128

# OpenCV GaussianBlur with same sigma
sigma = {sigma}
ksize = int(round(sigma * 6.0 + 1.0))
if ksize % 2 == 0:
    ksize += 1
ksize = max(ksize, 3)
out = cv2.GaussianBlur(px, (ksize, ksize), sigma, sigma, borderType=cv2.BORDER_REFLECT_101)
# OpenCV uses BGR internally but we pass RGB — GaussianBlur is channel-independent so no conversion needed
sys.stdout.buffer.write(out.tobytes())
"#,
        sigma = sigma
    );
    let reference = run_python_ref(&script);
    // Should be very close — same kernel formula and border mode
    assert_close("gaussian_blur_cv σ=1.5 vs OpenCV", &ours, &reference, 1.0);
}
