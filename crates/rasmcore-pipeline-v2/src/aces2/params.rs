//! ACES 2.0 test harness — reads binary reference vectors and validates our implementation.
//!
//! Binary format (from OCIO Docker generator):
//! Header (16 bytes): magic[4] "ACE2", count: u32, step_id: u32, reserved: u32
//! Per vector: input[3]: f32, output[3]: f32 (24 bytes per vector)

use std::path::Path;

/// Header of a binary reference file.
#[repr(C)]
#[derive(Debug)]
pub struct BinHeader {
    pub magic: [u8; 4],
    pub count: u32,
    pub step_id: u32,
    pub reserved: u32,
}

/// A single reference vector (input + expected output).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RefVector {
    pub input: [f32; 3],
    pub output: [f32; 3],
}

/// Load reference vectors from a binary file.
pub fn load_reference_vectors(path: &Path) -> Option<Vec<RefVector>> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 16 { return None; }

    let header: BinHeader = unsafe {
        std::ptr::read_unaligned(data.as_ptr() as *const BinHeader)
    };

    if &header.magic != b"ACE2" { return None; }

    let count = header.count as usize;
    let expected_size = 16 + count * 24;
    if data.len() < expected_size { return None; }

    let mut vectors = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 16 + i * 24;
        let v: RefVector = unsafe {
            std::ptr::read_unaligned(data[offset..].as_ptr() as *const RefVector)
        };
        vectors.push(v);
    }

    Some(vectors)
}

/// Find the reference vectors directory.
pub fn reference_dir() -> Option<std::path::PathBuf> {
    // Try relative to crate root (when running cargo test from workspace)
    let candidates = [
        "tests/aces2-vectors/output",
        "../../tests/aces2-vectors/output",
        "../../../tests/aces2-vectors/output",
    ];
    for c in &candidates {
        let p = Path::new(c);
        if p.exists() { return Some(p.to_path_buf()); }
    }
    None
}

/// Validate our implementation against reference vectors.
/// Returns (pass_count, fail_count, max_error) for reporting.
pub fn validate_against_reference(
    ref_path: &Path,
    tolerance: f32,
    transform_fn: impl Fn(&[f32; 3]) -> [f32; 3],
) -> (usize, usize, f32) {
    let vectors = match load_reference_vectors(ref_path) {
        Some(v) => v,
        None => {
            eprintln!("WARNING: Reference file not found: {:?}", ref_path);
            eprintln!("Run: ./tests/aces2-vectors/generate.sh to generate reference vectors");
            return (0, 0, 0.0);
        }
    };

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut max_err = 0.0f32;

    for (i, v) in vectors.iter().enumerate() {
        let actual = transform_fn(&v.input);
        let mut vec_err = 0.0f32;
        for c in 0..3 {
            let err = (actual[c] - v.output[c]).abs();
            vec_err = vec_err.max(err);
        }
        max_err = max_err.max(vec_err);

        if vec_err > tolerance {
            if fail < 10 { // only print first 10 failures
                eprintln!(
                    "  FAIL vec[{i}]: input=[{:.6}, {:.6}, {:.6}] expected=[{:.6}, {:.6}, {:.6}] got=[{:.6}, {:.6}, {:.6}] err={:.8}",
                    v.input[0], v.input[1], v.input[2],
                    v.output[0], v.output[1], v.output[2],
                    actual[0], actual[1], actual[2],
                    vec_err
                );
            }
            fail += 1;
        } else {
            pass += 1;
        }
    }

    (pass, fail, max_err)
}
