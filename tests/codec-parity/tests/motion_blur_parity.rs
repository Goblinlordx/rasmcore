//! Motion blur parity tests against OpenCV filter2D.
//!
//! Generates the same directional kernel in Rust and Python/OpenCV,
//! applies to the same test image, and compares pixel-by-pixel.
//! Tolerance: max +/-1 per channel for f32 rounding.
//!
//! Requires: tests/fixtures/.venv with opencv-python-headless + numpy.

use rasmcore_image::domain::filters::motion_blur;
use rasmcore_image::domain::types::{ColorSpace, ImageInfo, PixelFormat};
use std::process::Command;

fn venv_python() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures/.venv/bin/python3")
}

fn has_opencv() -> bool {
    Command::new(venv_python())
        .args(["-c", "import cv2"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Create a deterministic gradient test image (grayscale).
fn make_gradient_gray(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            pixels[(y * w + x) as usize] = ((x * 8 + y * 4) % 256) as u8;
        }
    }
    pixels
}

/// Build the same motion blur kernel that our Rust code builds.
fn build_kernel(length: u32, angle_degrees: f32) -> (Vec<f32>, usize) {
    let side = (2 * length + 1) as usize;
    let center = length as f32;
    let angle = angle_degrees.to_radians();
    let dx = angle.cos();
    let dy = -angle.sin();

    let mut kernel = vec![0.0f32; side * side];
    let steps = (length as f32 * 2.0).ceil() as usize * 2 + 1;
    let mut count = 0u32;
    for i in 0..steps {
        let t = (i as f32 / (steps - 1) as f32) * 2.0 - 1.0;
        let px = center + t * length as f32 * dx;
        let py = center + t * length as f32 * dy;
        let ix = px.round() as usize;
        let iy = py.round() as usize;
        if ix < side && iy < side {
            let idx = iy * side + ix;
            if kernel[idx] == 0.0 {
                kernel[idx] = 1.0;
                count += 1;
            }
        }
    }
    // Normalize
    if count > 0 {
        let c = count as f32;
        for v in &mut kernel {
            *v /= c;
        }
    }
    (kernel, side)
}

/// Run OpenCV filter2D with our exact kernel via Python.
/// This ensures kernel shape is identical — we only compare convolution output.
fn opencv_filter2d(pixels: &[u8], w: u32, h: u32, kernel: &[f32], kside: usize) -> Vec<u8> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = std::env::temp_dir();
    let input_path = tmp_dir.join(format!("_mblur_input_{id}.raw"));
    let kernel_path = tmp_dir.join(format!("_mblur_kernel_{id}.raw"));
    let output_path = tmp_dir.join(format!("_mblur_output_{id}.raw"));

    std::fs::write(&input_path, pixels).unwrap();
    // Write kernel as f32 little-endian bytes
    let kernel_bytes: Vec<u8> = kernel.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&kernel_path, &kernel_bytes).unwrap();

    let script = format!(
        r#"
import cv2, numpy as np
data = open('{input}', 'rb').read()
img = np.frombuffer(data, dtype=np.uint8).reshape({h}, {w})
kdata = open('{kpath}', 'rb').read()
kernel = np.frombuffer(kdata, dtype=np.float32).reshape({kside}, {kside})
result = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
open('{output}', 'wb').write(result.tobytes())
"#,
        input = input_path.display(),
        kpath = kernel_path.display(),
        output = output_path.display(),
        w = w,
        h = h,
        kside = kside,
    );

    let status = Command::new(venv_python())
        .args(["-c", &script])
        .status()
        .expect("failed to run Python");
    assert!(status.success(), "OpenCV Python script failed");

    std::fs::read(&output_path).unwrap()
}

fn compare_pixels(ours: &[u8], reference: &[u8], tolerance: u8) -> (u8, usize) {
    assert_eq!(ours.len(), reference.len());
    let mut max_diff: u8 = 0;
    let mut num_diff = 0;
    for (i, (&a, &b)) in ours.iter().zip(reference.iter()).enumerate() {
        let diff = (a as i16 - b as i16).unsigned_abs() as u8;
        if diff > 0 {
            num_diff += 1;
            if diff > max_diff {
                max_diff = diff;
                if diff > tolerance {
                    let x = i % 32;
                    let y = i / 32;
                    eprintln!(
                        "  MISMATCH at ({x},{y}): ours={a} ref={b} diff={diff}"
                    );
                }
            }
        }
    }
    (max_diff, num_diff)
}

macro_rules! motion_blur_test {
    ($name:ident, $length:expr, $angle:expr) => {
        #[test]
        fn $name() {
            if !has_opencv() {
                panic!(
                    "OpenCV not available. Install: tests/fixtures/.venv/bin/pip install \
                     opencv-python-headless numpy"
                );
            }

            let w = 32u32;
            let h = 32u32;
            let pixels = make_gradient_gray(w, h);
            let info = ImageInfo {
                width: w,
                height: h,
                format: PixelFormat::Gray8,
                color_space: ColorSpace::Srgb,
            };

            let ours = motion_blur(&pixels, &info, $length, $angle).unwrap();
            let (kernel, kside) = build_kernel($length, $angle);
            let reference = opencv_filter2d(&pixels, w, h, &kernel, kside);

            let (max_diff, num_diff) = compare_pixels(&ours, &reference, 1);
            eprintln!(
                "motion_blur(length={}, angle={}): max_diff={}, differing={}",
                $length, $angle, max_diff, num_diff
            );
            assert!(
                max_diff <= 1,
                "motion_blur(length={}, angle={}) FAILED: max_diff={} exceeds tolerance 1",
                $length, $angle, max_diff
            );
        }
    };
}

motion_blur_test!(parity_horizontal_3, 3, 0.0);
motion_blur_test!(parity_horizontal_5, 5, 0.0);
motion_blur_test!(parity_vertical_3, 3, 90.0);
motion_blur_test!(parity_vertical_5, 5, 90.0);
motion_blur_test!(parity_diagonal_45_3, 3, 45.0);
motion_blur_test!(parity_diagonal_45_5, 5, 45.0);
motion_blur_test!(parity_diagonal_135_3, 3, 135.0);
motion_blur_test!(parity_30_degrees, 4, 30.0);
motion_blur_test!(parity_60_degrees, 4, 60.0);
motion_blur_test!(parity_120_degrees, 3, 120.0);
