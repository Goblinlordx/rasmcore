//! Fusion helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Compute Mertens weight map for a single image.
/// Input is f32 RGB in [0,1], interleaved. Returns one weight per pixel.
pub fn compute_mertens_weight(img_f: &[f32], w: usize, h: usize, params: &MertensParams) -> Vec<f32> {
    let n = w * h;
    let _sigma = 0.2f32;

    // Convert to grayscale — matches OpenCV MergeMertens which uses COLOR_RGB2GRAY
    // on BGR data, effectively: 0.299*B + 0.587*G + 0.114*R.
    // Our input is RGB, so: 0.114*R + 0.587*G + 0.299*B
    let mut gray = vec![0.0f32; n];
    for i in 0..n {
        let r = img_f[i * 3];
        let g = img_f[i * 3 + 1];
        let b = img_f[i * 3 + 2];
        gray[i] = 0.114 * r + 0.587 * g + 0.299 * b;
    }

    // Contrast: abs(Laplacian(gray)) — standard 3×3 kernel [[0,1,0],[1,-4,1],[0,1,0]]
    // Border: BORDER_REFLECT_101 (OpenCV BORDER_DEFAULT)
    let mut contrast = vec![0.0f32; n];
    let ws = w as isize;
    let hs = h as isize;
    for y in 0..h {
        for x in 0..w {
            let yp = reflect101(y as isize - 1, hs) as usize;
            let yn = reflect101(y as isize + 1, hs) as usize;
            let xp = reflect101(x as isize - 1, ws) as usize;
            let xn = reflect101(x as isize + 1, ws) as usize;

            let center = gray[y * w + x];
            let lap = gray[yp * w + x] + gray[yn * w + x] + gray[y * w + xp] + gray[y * w + xn]
                - 4.0 * center;
            contrast[y * w + x] = lap.abs();
        }
    }

    // Saturation: sqrt(sum((ch - mean)²)) — matches OpenCV MergeMertens exactly.
    // Note: OpenCV does NOT divide by channel count before sqrt (not population std).
    let mut saturation = vec![0.0f32; n];
    for i in 0..n {
        let r = img_f[i * 3];
        let g = img_f[i * 3 + 1];
        let b = img_f[i * 3 + 2];
        let mu = (r + g + b) / 3.0;
        let sum_sq = (r - mu) * (r - mu) + (g - mu) * (g - mu) + (b - mu) * (b - mu);
        saturation[i] = sum_sq.sqrt();
    }

    // Well-exposedness: product over channels of exp(-(ch - 0.5)² / (2 * σ²))
    // OpenCV computes: expo = (ch - 0.5)²; expo = -expo / 0.08; exp(expo)
    // where 0.08 = 2 * 0.2² = 2 * σ². Match the exact operation order.
    let mut well_exp = vec![1.0f32; n];
    for i in 0..n {
        for c in 0..3 {
            let ch = img_f[i * 3 + c];
            let expo = ch - 0.5;
            let expo = expo * expo;
            let expo = -expo / 0.08;
            well_exp[i] *= expo.exp();
        }
    }

    // Combined weight
    let mut weight = vec![0.0f32; n];
    for i in 0..n {
        let mut w = 1.0f32;
        if params.contrast_weight != 0.0 {
            w *= contrast[i].powf(params.contrast_weight);
        }
        if params.saturation_weight != 0.0 {
            w *= saturation[i].powf(params.saturation_weight);
        }
        if params.exposure_weight != 0.0 {
            w *= well_exp[i].powf(params.exposure_weight);
        }
        weight[i] = w + 1e-12; // avoid zero weights
    }

    weight
}

/// Debevec HDR merge — compute radiance map from bracketed exposures + response curve.
///
/// Returns f32 HDR radiance map (3-channel interleaved, linear values).
pub fn debevec_hdr_merge(
    images: &[&[u8]],
    info: &ImageInfo,
    exposure_times: &[f32],
    response: &[[f32; 256]],
) -> Result<Vec<f32>, ImageError> {
    if images.len() < 2 || images.len() != exposure_times.len() {
        return Err(ImageError::InvalidInput(
            "need matching images and exposure times".into(),
        ));
    }
    if response.len() != 3 {
        return Err(ImageError::InvalidInput(
            "response must have 3 channels".into(),
        ));
    }
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "debevec requires Rgb8 input".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let n = w * h;

    let mut hdr = vec![0.0f32; n * 3];

    for i in 0..n {
        for ch in 0..3 {
            let mut num = 0.0f64;
            let mut den = 0.0f64;

            for (img, &dt) in images.iter().zip(exposure_times.iter()) {
                let z = img[i * 3 + ch] as usize;
                let wt = hat_weight(z);
                let ln_dt = (dt as f64).ln();
                num += wt * (response[ch][z] as f64 - ln_dt);
                den += wt;
            }

            hdr[i * 3 + ch] = if den > 0.0 {
                (num / den).exp() as f32
            } else {
                0.0
            };
        }
    }

    Ok(hdr)
}

/// Estimate camera response curve using Debevec & Malik's method.
///
/// Takes bracketed exposures (u8 images) and exposure times.
/// Returns 256-entry response curve per channel (natural log of exposure).
///
/// Reference: Debevec & Malik "Recovering High Dynamic Range Radiance Maps
/// from Photographs" (SIGGRAPH 1997).
/// Matches OpenCV cv2.createCalibrateDebevec.
pub fn debevec_response_curve(
    images: &[&[u8]],
    info: &ImageInfo,
    exposure_times: &[f32],
    params: &DebevecParams,
) -> Result<Vec<[f32; 256]>, ImageError> {
    if images.len() < 2 || images.len() != exposure_times.len() {
        return Err(ImageError::InvalidInput(
            "need matching images and exposure times".into(),
        ));
    }
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "debevec requires Rgb8 input".into(),
        ));
    }

    let (w, h) = (info.width as usize, info.height as usize);
    let n = w * h;
    let n_images = images.len();

    // Select sample pixels (deterministic, evenly spaced)
    let n_samples = params.samples.min(n);
    let step = n / n_samples;
    let sample_indices: Vec<usize> = (0..n_samples).map(|i| i * step).collect();

    let channels = 3;
    let mut response_curves = Vec::with_capacity(channels);

    for ch in 0..channels {
        // Solve for response curve g(z) where g(z) = ln(exposure)
        // Using SVD-based least squares from Debevec paper
        let n_eq = n_samples * n_images + 256 + 1; // data + smoothness + constraint
        let n_unknowns = 256 + n_samples; // g(0..255) + ln(E_i)

        // Build overdetermined system A*x = b
        let mut a = vec![0.0f64; n_eq * n_unknowns];
        let mut b = vec![0.0f64; n_eq];

        let mut eq = 0;

        // Data equations: w(z) * [g(z) - ln(dt) - ln(E)] = 0
        for (s, &si) in sample_indices.iter().enumerate() {
            for (img, &dt) in images.iter().zip(exposure_times.iter()) {
                let z = img[si * 3 + ch] as usize;
                let wt = hat_weight(z);

                a[eq * n_unknowns + z] = wt; // g(z) coefficient
                a[eq * n_unknowns + 256 + s] = -wt; // -ln(E_i) coefficient
                b[eq] = wt * (dt as f64).ln(); // w(z) * ln(dt)
                eq += 1;
            }
        }

        // Smoothness equations: lambda * w(z) * [g(z-1) - 2*g(z) + g(z+1)] = 0
        let lam = params.lambda as f64;
        for z in 1..255 {
            let wt = hat_weight(z);
            a[eq * n_unknowns + (z - 1)] = lam * wt;
            a[eq * n_unknowns + z] = -2.0 * lam * wt;
            a[eq * n_unknowns + (z + 1)] = lam * wt;
            b[eq] = 0.0;
            eq += 1;
        }

        // Fix g(128) = 0 (constraint for midpoint)
        a[eq * n_unknowns + 128] = 1.0;
        b[eq] = 0.0;
        eq += 1;

        // Solve via normal equations: A^T A x = A^T b
        let x = solve_least_squares(&a, &b, eq, n_unknowns);

        let mut curve = [0.0f32; 256];
        for z in 0..256 {
            curve[z] = x[z] as f32;
        }
        response_curves.push(curve);
    }

    Ok(response_curves)
}

/// Mertens exposure fusion — blends multiple exposures without HDR intermediate.
///
/// Takes a list of same-size RGB8 images and produces a fused result.
/// Uses Laplacian pyramid blending with per-pixel weights based on
/// contrast, saturation, and well-exposedness.
///
/// Reference: OpenCV cv2.createMergeMertens (photo/src/merge.cpp).
/// Algorithm: Mertens et al. "Exposure Fusion" (Pacific Graphics 2007).
pub fn mertens_fusion(
    images: &[&[u8]],
    info: &ImageInfo,
    params: &MertensParams,
) -> Result<Vec<u8>, ImageError> {
    if images.len() < 2 {
        return Err(ImageError::InvalidInput("need at least 2 images".into()));
    }
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "mertens fusion requires Rgb8 input".into(),
        ));
    }
    let (w, h) = (info.width as usize, info.height as usize);
    let n = w * h;
    let expected_len = n * 3;
    for img in images {
        if img.len() != expected_len {
            return Err(ImageError::InvalidInput("image size mismatch".into()));
        }
    }

    let n_images = images.len();

    // Convert images to f32 [0,1] (3-channel interleaved)
    let images_f: Vec<Vec<f32>> = images
        .iter()
        .map(|img| img.iter().map(|&v| v as f32 / 255.0).collect())
        .collect();

    // Step 1: Compute per-pixel weights for each image
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n_images);
    for img_f in &images_f {
        let weight = compute_mertens_weight(img_f, w, h, params);
        weights.push(weight);
    }

    // Step 2: Normalize weights (sum to 1 per pixel)
    for px in 0..n {
        let sum: f32 = weights.iter().map(|w| w[px]).sum();
        if sum > 0.0 {
            for w in &mut weights {
                w[px] /= sum;
            }
        }
    }

    // Step 3: Laplacian pyramid blending
    // Pyramid depth: log2(min(w,h))
    // Match OpenCV: int(logf(float(min(w,h))) / logf(2.0f))
    let maxlevel = ((w.min(h) as f32).ln() / 2.0f32.ln()) as usize;

    // Build weight Gaussian pyramids and image Laplacian pyramids
    let weight_pyrs: Vec<Vec<Vec<f32>>> = weights
        .iter()
        .map(|w| gaussian_pyramid_gray(w, info.width, info.height, maxlevel))
        .collect();

    let image_lap_pyrs: Vec<Vec<(Vec<f32>, u32, u32)>> = images_f
        .iter()
        .map(|img| laplacian_pyramid_rgb(img, info.width, info.height, maxlevel))
        .collect();

    // Step 4: Merge at each level
    let mut merged_pyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(maxlevel + 1);
    for level in 0..=maxlevel {
        let (_, lw, lh) = image_lap_pyrs[0][level];
        let lpx = (lw * lh) as usize;
        let mut merged = vec![0.0f32; lpx * 3];

        for i in 0..n_images {
            let (ref lap, _, _) = image_lap_pyrs[i][level];
            let weight = &weight_pyrs[i][level];
            for px in 0..lpx {
                let wt = weight[px];
                merged[px * 3] += wt * lap[px * 3];
                merged[px * 3 + 1] += wt * lap[px * 3 + 1];
                merged[px * 3 + 2] += wt * lap[px * 3 + 2];
            }
        }
        merged_pyr.push((merged, lw, lh));
    }

    // Step 5: Collapse the merged Laplacian pyramid
    let (mut result, mut rw, mut rh) = merged_pyr.pop().unwrap();
    for level in (0..maxlevel).rev() {
        let (ref lap, lw, lh) = merged_pyr[level];
        let upsampled = pyr_up_rgb(&result, rw, rh, lw, lh);
        result = Vec::with_capacity((lw * lh) as usize * 3);
        let lpx = (lw * lh) as usize;
        for px in 0..(lpx * 3) {
            result.push(upsampled[px] + lap[px]);
        }
        rw = lw;
        rh = lh;
    }

    // Convert back to u8, clamp
    let mut output = vec![0u8; n * 3];
    for i in 0..(n * 3) {
        output[i] = (result[i] * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(output)
}

/// Mertens fusion returning f32 output in [0,1] range (for precision testing).
pub fn mertens_fusion_f32(
    images: &[&[u8]],
    info: &ImageInfo,
    params: &MertensParams,
) -> Result<Vec<f32>, ImageError> {
    if images.len() < 2 {
        return Err(ImageError::InvalidInput("need at least 2 images".into()));
    }
    if info.format != PixelFormat::Rgb8 {
        return Err(ImageError::UnsupportedFormat(
            "mertens fusion requires Rgb8 input".into(),
        ));
    }
    let (w, h) = (info.width as usize, info.height as usize);
    let n = w * h;
    let expected_len = n * 3;
    for img in images {
        if img.len() != expected_len {
            return Err(ImageError::InvalidInput("image size mismatch".into()));
        }
    }

    let n_images = images.len();
    let images_f: Vec<Vec<f32>> = images
        .iter()
        .map(|img| img.iter().map(|&v| v as f32 / 255.0).collect())
        .collect();

    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n_images);
    for img_f in &images_f {
        let weight = compute_mertens_weight(img_f, w, h, params);
        weights.push(weight);
    }

    for px in 0..n {
        let sum: f32 = weights.iter().map(|w| w[px]).sum();
        if sum > 0.0 {
            for w in &mut weights {
                w[px] /= sum;
            }
        }
    }

    // Match OpenCV: int(logf(float(min(w,h))) / logf(2.0f))
    let maxlevel = ((w.min(h) as f32).ln() / 2.0f32.ln()) as usize;

    let weight_pyrs: Vec<Vec<Vec<f32>>> = weights
        .iter()
        .map(|w| gaussian_pyramid_gray(w, info.width, info.height, maxlevel))
        .collect();

    let image_lap_pyrs: Vec<Vec<(Vec<f32>, u32, u32)>> = images_f
        .iter()
        .map(|img| laplacian_pyramid_rgb(img, info.width, info.height, maxlevel))
        .collect();

    let mut merged_pyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(maxlevel + 1);
    for level in 0..=maxlevel {
        let (_, lw, lh) = image_lap_pyrs[0][level];
        let lpx = (lw * lh) as usize;
        let mut merged = vec![0.0f32; lpx * 3];

        for i in 0..n_images {
            let (ref lap, _, _) = image_lap_pyrs[i][level];
            let weight = &weight_pyrs[i][level];
            for px in 0..lpx {
                let wt = weight[px];
                merged[px * 3] += wt * lap[px * 3];
                merged[px * 3 + 1] += wt * lap[px * 3 + 1];
                merged[px * 3 + 2] += wt * lap[px * 3 + 2];
            }
        }
        merged_pyr.push((merged, lw, lh));
    }

    let (mut result, mut rw, mut rh) = merged_pyr.pop().unwrap();
    for level in (0..maxlevel).rev() {
        let (ref lap, lw, lh) = merged_pyr[level];
        let upsampled = pyr_up_rgb(&result, rw, rh, lw, lh);
        result = Vec::with_capacity((lw * lh) as usize * 3);
        let lpx = (lw * lh) as usize;
        for px in 0..(lpx * 3) {
            result.push(upsampled[px] + lap[px]);
        }
        rw = lw;
        rh = lh;
    }

    Ok(result)
}

