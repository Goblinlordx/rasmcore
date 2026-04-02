//! name: invert_compute
//! category: adjustment
//! strategy: compute

// Compute strategy invert: iterates over all bytes and inverts each.
// Self-inverse: applying twice returns the original image.
// Uses Rhai Blob (Vec<u8>) for direct pixel manipulation.

fn compute(pixels, info, params) {
    let out = pixels;
    for i in range(0, out.len()) {
        out[i] = 255 - out[i];
    }
    out
}
