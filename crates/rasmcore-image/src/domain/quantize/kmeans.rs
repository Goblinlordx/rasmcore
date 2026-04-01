// ─── K-Means Color Clustering ─────────────────────────────────────────────
//
// Standard Lloyd's algorithm for color palette extraction.
// NOT validated against ImageMagick or other reference implementation.
// Property-tested only (determinism, convergence, identity).

use super::{ImageError, ImageInfo, Rgb};

/// Extract a k-color palette from an RGB8 image using Lloyd's k-means algorithm.
///
/// - `k`: number of clusters (2..=256)
/// - `max_iterations`: convergence limit (typically 20-50)
/// - `seed`: deterministic PRNG seed for reproducibility
///
/// For images larger than 1M pixels, subsamples to ~250K pixels for clustering
/// then returns the final centroids.
pub fn kmeans_palette(
    pixels: &[u8],
    info: &ImageInfo,
    k: usize,
    max_iterations: u32,
    seed: u64,
) -> Result<Vec<Rgb>, ImageError> {
    if !(2..=256).contains(&k) {
        return Err(ImageError::InvalidParameters("k must be 2..256".into()));
    }
    let total_pixels = (info.width * info.height) as usize;
    if pixels.len() < total_pixels * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }

    // Collect RGB triplets, subsampling for large images
    let step = if total_pixels > 1_000_000 {
        (total_pixels / 250_000).max(2)
    } else {
        1
    };
    let mut samples: Vec<[u8; 3]> = Vec::with_capacity(total_pixels / step + 1);
    for i in (0..total_pixels).step_by(step) {
        samples.push([pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]]);
    }

    if samples.len() < k {
        return Err(ImageError::InvalidParameters(
            "not enough unique pixels for k clusters".into(),
        ));
    }

    // Initialize centroids: pick k pixels using a simple LCG PRNG
    let mut rng_state = seed.wrapping_add(1); // avoid seed=0 degenerate case
    let mut centroids: Vec<[f64; 3]> = Vec::with_capacity(k);
    let mut used = std::collections::HashSet::new();
    for _ in 0..k {
        loop {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (rng_state >> 33) as usize % samples.len();
            if used.insert(idx) {
                let s = samples[idx];
                centroids.push([s[0] as f64, s[1] as f64, s[2] as f64]);
                break;
            }
            if used.len() >= samples.len() {
                // Not enough unique pixels — just duplicate
                let s = samples[idx];
                centroids.push([s[0] as f64, s[1] as f64, s[2] as f64]);
                break;
            }
        }
    }

    // Lloyd's iterations
    let mut assignments = vec![0u32; samples.len()];
    let convergence_threshold = 0.5; // sub-pixel centroid movement

    for _ in 0..max_iterations {
        // Assignment step: assign each sample to nearest centroid
        for (i, s) in samples.iter().enumerate() {
            let (r, g, b) = (s[0] as f64, s[1] as f64, s[2] as f64);
            let mut best = 0u32;
            let mut best_dist = f64::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let dr = r - c[0];
                let dg = g - c[1];
                let db = b - c[2];
                let dist = dr * dr + dg * dg + db * db;
                if dist < best_dist {
                    best_dist = dist;
                    best = ci as u32;
                }
            }
            assignments[i] = best;
        }

        // Update step: recompute centroids as mean of assigned samples
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0u64; k];
        for (i, s) in samples.iter().enumerate() {
            let ci = assignments[i] as usize;
            sums[ci][0] += s[0] as f64;
            sums[ci][1] += s[1] as f64;
            sums[ci][2] += s[2] as f64;
            counts[ci] += 1;
        }

        let mut max_movement = 0.0f64;
        for ci in 0..k {
            if counts[ci] == 0 {
                continue; // empty cluster — keep old centroid
            }
            let new_r = sums[ci][0] / counts[ci] as f64;
            let new_g = sums[ci][1] / counts[ci] as f64;
            let new_b = sums[ci][2] / counts[ci] as f64;
            let dr = new_r - centroids[ci][0];
            let dg = new_g - centroids[ci][1];
            let db = new_b - centroids[ci][2];
            let movement = dr * dr + dg * dg + db * db;
            if movement > max_movement {
                max_movement = movement;
            }
            centroids[ci] = [new_r, new_g, new_b];
        }

        if max_movement < convergence_threshold {
            break;
        }
    }

    // Convert centroids to Rgb
    Ok(centroids
        .iter()
        .map(|c| Rgb {
            r: c[0].round().clamp(0.0, 255.0) as u8,
            g: c[1].round().clamp(0.0, 255.0) as u8,
            b: c[2].round().clamp(0.0, 255.0) as u8,
        })
        .collect())
}
