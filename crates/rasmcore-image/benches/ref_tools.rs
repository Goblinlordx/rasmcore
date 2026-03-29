//! Helpers for invoking reference tools (ImageMagick, libvips, cwebp, etc.) in benchmarks.

use std::process::Command;

/// Check if a command-line tool is available.
pub fn has_tool(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a command and return (stdout bytes, elapsed time).
/// Panics if the command fails.
pub fn run_timed(cmd: &str, args: &[&str]) -> (Vec<u8>, std::time::Duration) {
    let start = std::time::Instant::now();
    let output = Command::new(cmd)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("{cmd} failed to start: {e}"));
    let elapsed = start.elapsed();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("{cmd} failed: {stderr}");
    }
    (output.stdout, elapsed)
}

/// Run a command writing stdin and capturing stdout.
pub fn run_with_stdin(cmd: &str, args: &[&str], input: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("{cmd} failed to start: {e}"));
    child.stdin.take().unwrap().write_all(input).unwrap();
    let output = child.wait_with_output().unwrap();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("{cmd} failed: {stderr}");
    }
    output.stdout
}

// ─── ImageMagick ─────────────────────────────────────────────────────────

/// Decode an image file using ImageMagick (magick convert input ppm:-).
pub fn magick_decode(input_path: &str) -> Vec<u8> {
    run_with_stdin("magick", &["convert", input_path, "ppm:-"], &[])
}

/// Encode raw pixels to a format via ImageMagick.
/// Writes to a temp file and reads back.
pub fn magick_encode(input_path: &str, output_fmt: &str, quality: Option<u8>) -> Vec<u8> {
    let tmp = std::env::temp_dir().join(format!("rasmcore_bench_out.{output_fmt}"));
    let tmp_str = tmp.to_str().unwrap();
    let mut args = vec!["convert", input_path];
    let q_str;
    if let Some(q) = quality {
        q_str = q.to_string();
        args.extend(&["-quality", &q_str]);
    }
    args.push(tmp_str);
    let _ = Command::new("magick").args(&args).output().unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}

/// Run a pipeline of operations via ImageMagick.
pub fn magick_pipeline(
    input_path: &str,
    ops: &[&str],
    output_fmt: &str,
    quality: Option<u8>,
) -> Vec<u8> {
    let tmp = std::env::temp_dir().join(format!("rasmcore_bench_pipe.{output_fmt}"));
    let tmp_str = tmp.to_str().unwrap();
    let mut args: Vec<&str> = vec!["convert", input_path];
    args.extend_from_slice(ops);
    let q_str;
    if let Some(q) = quality {
        q_str = q.to_string();
        args.extend(&["-quality", &q_str]);
    }
    args.push(tmp_str);
    let _ = Command::new("magick").args(&args).output().unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}

// ─── libvips ─────────────────────────────────────────────────────────────

/// Run a pipeline of operations via vips CLI.
pub fn vips_pipeline(
    input_path: &str,
    ops: &str,
    output_fmt: &str,
    quality: Option<u8>,
) -> Vec<u8> {
    let tmp = std::env::temp_dir().join(format!("rasmcore_bench_vips.{output_fmt}"));
    let tmp_str = tmp.to_str().unwrap();
    // vips uses: vips <operation> <input> <output> [params...]
    // For pipelines, we use vipsthumbnail or chain commands
    // Simplest approach: use vips copy for decode, or the vips CLI directly
    let q_arg;
    let mut args: Vec<&str> = vec![input_path, tmp_str];
    if let Some(q) = quality {
        q_arg = format!("[Q={q}]");
        // vips embeds quality in the output filename: out.jpg[Q=85]
        let tmp_q = std::env::temp_dir().join(format!("rasmcore_bench_vips.{output_fmt}"));
        let tmp_q_str = format!("{}{}", tmp_q.to_str().unwrap(), q_arg);
        let _ = Command::new("vips")
            .args(&[ops, input_path, &tmp_q_str])
            .output()
            .unwrap();
        return std::fs::read(&tmp).unwrap_or_default();
    }
    let _ = Command::new("vips")
        .args(&[ops, input_path, tmp_str])
        .output()
        .unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}

/// Decode using vips (vips copy input.jpg output.ppm).
pub fn vips_decode(input_path: &str) -> Vec<u8> {
    let tmp = std::env::temp_dir().join("rasmcore_bench_vips_dec.ppm");
    let tmp_str = tmp.to_str().unwrap();
    let _ = Command::new("vips")
        .args(&["copy", input_path, tmp_str])
        .output()
        .unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}

/// Encode using vips.
pub fn vips_encode(input_path: &str, output_fmt: &str, quality: Option<u8>) -> Vec<u8> {
    let tmp = std::env::temp_dir().join(format!("rasmcore_bench_vips_enc.{output_fmt}"));
    let tmp_path = if let Some(q) = quality {
        format!("{}[Q={q}]", tmp.to_str().unwrap())
    } else {
        tmp.to_str().unwrap().to_string()
    };
    let _ = Command::new("vips")
        .args(&["copy", input_path, &tmp_path])
        .output()
        .unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}

// ─── cwebp/dwebp ────────────────────────────────────────────────────────

pub fn cwebp_encode(input_path: &str, quality: u8) -> Vec<u8> {
    let tmp = std::env::temp_dir().join("rasmcore_bench_cwebp.webp");
    let tmp_str = tmp.to_str().unwrap();
    let q = quality.to_string();
    let _ = Command::new("cwebp")
        .args(&["-q", &q, input_path, "-o", tmp_str])
        .output()
        .unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}

/// Decode WebP using dwebp.
pub fn dwebp_decode(input_path: &str) -> Vec<u8> {
    let tmp = std::env::temp_dir().join("rasmcore_bench_dwebp.ppm");
    let tmp_str = tmp.to_str().unwrap();
    let _ = Command::new("dwebp")
        .args(&[input_path, "-o", tmp_str])
        .output()
        .unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}

// ─── libjpeg-turbo ──────────────────────────────────────────────────────

pub fn cjpeg_encode(input_path: &str, quality: u8) -> Vec<u8> {
    let tmp = std::env::temp_dir().join("rasmcore_bench_cjpeg.jpg");
    let tmp_str = tmp.to_str().unwrap();
    let q = quality.to_string();
    let _ = Command::new("cjpeg")
        .args(&["-quality", &q, "-outfile", tmp_str, input_path])
        .output()
        .unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}

pub fn djpeg_decode(input_path: &str) -> Vec<u8> {
    let tmp = std::env::temp_dir().join("rasmcore_bench_djpeg.ppm");
    let tmp_str = tmp.to_str().unwrap();
    let _ = Command::new("djpeg")
        .args(&["-outfile", tmp_str, input_path])
        .output()
        .unwrap();
    std::fs::read(&tmp).unwrap_or_default()
}
