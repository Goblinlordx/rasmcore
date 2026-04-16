//! Reference implementations of distortion/warp operations.
//!
//! Pure math, no dependencies. All functions operate on `&[f32]` RGBA
//! interleaved buffers in linear f32 space. Alpha is preserved (sampled
//! identically to color channels). Every distortion computes source
//! coordinates for each output pixel and bilinear-samples the input.

// ─── Bilinear Sampling Helper ──────────────────────────────────────────────

/// Bilinear sample from an RGBA interleaved buffer, clamping at edges.
#[inline]
fn bilinear_sample(input: &[f32], w: u32, h: u32, x: f32, y: f32) -> [f32; 4] {
    let w = w as usize;
    let h = h as usize;

    let x0 = (x.floor() as isize).clamp(0, w as isize - 1) as usize;
    let y0 = (y.floor() as isize).clamp(0, h as isize - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = (x - x.floor()).clamp(0.0, 1.0);
    let fy = (y - y.floor()).clamp(0.0, 1.0);

    let idx = |px: usize, py: usize| (py * w + px) * 4;

    let i00 = idx(x0, y0);
    let i10 = idx(x1, y0);
    let i01 = idx(x0, y1);
    let i11 = idx(x1, y1);

    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let v00 = input[i00 + c];
        let v10 = input[i10 + c];
        let v01 = input[i01 + c];
        let v11 = input[i11 + c];
        out[c] = v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy;
    }
    out
}

// ─── 1. Barrel / Pincushion ────────────────────────────────────────────────

/// Barrel (k1>0) or pincushion (k1<0) distortion.
///
/// Normalizes pixel coordinates to [-1, 1], computes radial distortion
/// r' = r * (1 + k1*r² + k2*r⁴), then denormalizes and bilinear samples.
pub fn barrel(input: &[f32], w: u32, h: u32, k1: f32, k2: f32) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    // Normalize so that corners map to sqrt(2) approximately
    let norm = cx.max(cy).max(1.0);

    for y in 0..h {
        for x in 0..w {
            let nx = (x as f32 - cx) / norm;
            let ny = (y as f32 - cy) / norm;
            let r2 = nx * nx + ny * ny;
            let r4 = r2 * r2;
            let scale = 1.0 + k1 * r2 + k2 * r4;
            let sx = nx * scale * norm + cx;
            let sy = ny * scale * norm + cy;
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (y * w + x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 2. Spherize ───────────────────────────────────────────────────────────

/// Spherize distortion — maps through a sphere projection.
///
/// Normalizes to [-1, 1] independently per axis (elliptical): nx=(x-cx)/cx.
/// cx = w/2, cy = h/2. Inside the unit ellipse applies asin-based correction:
///   theta = asin(r) / r; factor = 1 + amount * (theta - 1)
pub fn spherize(input: &[f32], w: u32, h: u32, amount: f32) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;

    for y in 0..h {
        for x in 0..w {
            let nx = if cx > 0.0 { (x as f32 - cx) / cx } else { 0.0 };
            let ny = if cy > 0.0 { (y as f32 - cy) / cy } else { 0.0 };
            let r = (nx * nx + ny * ny).sqrt();
            let (sx, sy) = if r > 0.0 && r < 1.0 {
                let theta = r.asin() / r;
                let factor = 1.0 + amount * (theta - 1.0);
                (nx * factor * cx + cx, ny * factor * cy + cy)
            } else {
                (x as f32, y as f32)
            };
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (y * w + x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 3. Swirl ──────────────────────────────────────────────────────────────

/// Swirl distortion — rotates pixels around center.
///
/// Rotation angle decreases quadratically with distance from center:
/// t = 1 - dist/radius; theta = angle * t * t for dist < radius.
pub fn swirl(input: &[f32], w: u32, h: u32, angle: f32, radius: f32) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];
    let cx = (w as f32 - 1.0) / 2.0;
    let cy = (h as f32 - 1.0) / 2.0;

    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let (sx, sy) = if dist < radius && radius > 0.0 {
                let t = 1.0 - dist / radius;
                let theta = angle * t * t;
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                (
                    cx + dx * cos_t - dy * sin_t,
                    cy + dx * sin_t + dy * cos_t,
                )
            } else {
                (x as f32, y as f32)
            };
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (y * w + x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 4. Wave ───────────────────────────────────────────────────────────────

/// Sinusoidal wave displacement.
///
/// If `horizontal`: each pixel's y-source is displaced by
/// `amplitude * sin(2π * x / wavelength)`.
/// Otherwise: x-source displaced by `amplitude * sin(2π * y / wavelength)`.
pub fn wave(
    input: &[f32],
    w: u32,
    h: u32,
    amplitude: f32,
    wavelength: f32,
    horizontal: bool,
) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];
    let two_pi = 2.0 * std::f32::consts::PI;

    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = if horizontal {
                let dy = amplitude * (two_pi * x as f32 / wavelength).sin();
                (x as f32, y as f32 + dy)
            } else {
                let dx = amplitude * (two_pi * y as f32 / wavelength).sin();
                (x as f32 + dx, y as f32)
            };
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (y * w + x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 5. Ripple ─────────────────────────────────────────────────────────────

/// Radial ripple from center.
///
/// Offset = amplitude * sin(2π * dist / wavelength), applied radially.
pub fn ripple(input: &[f32], w: u32, h: u32, amplitude: f32, wavelength: f32) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let two_pi = 2.0 * std::f32::consts::PI;

    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let (sx, sy) = if dist > 0.0 {
                let offset = amplitude * (two_pi * dist / wavelength).sin();
                let nx = dx / dist;
                let ny = dy / dist;
                (x as f32 + nx * offset, y as f32 + ny * offset)
            } else {
                (x as f32, y as f32)
            };
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (y * w + x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 6. Polar (Cartesian → Polar) ─────────────────────────────────────────

/// Cartesian to polar coordinate transform.
///
/// Output pixel (out_x, out_y) maps:
/// - out_y corresponds to radius (0 = center, h = max radius)
/// - out_x corresponds to angle (0 = 0°, w = 360°)
///
/// max_radius = min(cx, cy). Angle: (x+0.5-cx)/w*2PI.
/// Radius: (y+0.5)/h * max_radius. Maps via sin(angle)->x, cos(angle)->y.
pub fn polar(input: &[f32], w: u32, h: u32) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_radius = cx.min(cy);
    let two_pi = 2.0 * std::f32::consts::PI;

    for out_y in 0..h {
        for out_x in 0..w {
            // out_x -> angle, out_y -> radius
            let angle = (out_x as f32 + 0.5 - cx) / w as f32 * two_pi;
            let radius = (out_y as f32 + 0.5) / h as f32 * max_radius;
            let sx = cx + radius * angle.sin() - 0.5;
            let sy = cy + radius * angle.cos() - 0.5;
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (out_y * w + out_x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 7. Depolar (Polar → Cartesian) ───────────────────────────────────────

/// Polar to cartesian coordinate transform (inverse of `polar`).
///
/// For each output (out_x, out_y) in cartesian space, compute the
/// corresponding polar coordinates and sample the polar-space input.
/// Uses atan2(dx, dy) (sin→x, cos→y convention) matching polar().
pub fn depolar(input: &[f32], w: u32, h: u32) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_radius = cx.min(cy);
    let two_pi = 2.0 * std::f32::consts::PI;

    for out_y in 0..h {
        for out_x in 0..w {
            let dx = out_x as f32 + 0.5 - cx;
            let dy = out_y as f32 + 0.5 - cy;
            let radius = (dx * dx + dy * dy).sqrt();
            // atan2(dx, dy): angle from +Y axis going CW (matches sin→x, cos→y in polar)
            let mut angle = dx.atan2(dy);
            if angle < 0.0 {
                angle += two_pi;
            }
            // Map back to polar image coordinates
            let mut xx = angle / two_pi;
            xx -= xx.round();
            let sx = xx * w as f32 + cx - 0.5;
            let sy = radius * (h as f32 / max_radius) - 0.5;
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (out_y * w + out_x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 8. Displacement Map ───────────────────────────────────────────────────

/// Displacement map distortion.
///
/// Uses R channel of `map` as dx, G channel as dy (centered at 0.5):
/// sample input at (x + (map_r - 0.5) * scale_x, y + (map_g - 0.5) * scale_y).
pub fn displacement_map(
    input: &[f32],
    w: u32,
    h: u32,
    map: &[f32],
    scale_x: f32,
    scale_y: f32,
) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];

    for y in 0..h {
        for x in 0..w {
            let map_idx = (y * w + x) as usize * 4;
            let map_r = map[map_idx];
            let map_g = map[map_idx + 1];
            let sx = x as f32 + (map_r - 0.5) * scale_x;
            let sy = y as f32 + (map_g - 0.5) * scale_y;
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (y * w + x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 9. Liquify ────────────────────────────────────────────────────────────

/// Liquify — push pixels away from a center point.
///
/// Within `radius` of (cx, cy): displacement = (dx, dy) * (1 - dist/radius)².
/// Pixels outside radius are unchanged.
pub fn liquify(
    input: &[f32],
    w: u32,
    h: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    dx: f32,
    dy: f32,
) -> Vec<f32> {
    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];

    for y in 0..h {
        for x in 0..w {
            let px_x = x as f32;
            let px_y = y as f32;
            let ddx = px_x - cx;
            let ddy = px_y - cy;
            let dist = (ddx * ddx + ddy * ddy).sqrt();
            let (sx, sy) = if dist < radius && radius > 0.0 {
                let t = dist / radius;
                let w = (-2.0 * t * t).exp();
                (px_x - dx * w, px_y - dy * w)
            } else {
                (px_x, px_y)
            };
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (y * w + x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── 10. Mesh Warp ─────────────────────────────────────────────────────────

/// Mesh warp — bilinear interpolation of displacement vectors from a control grid.
///
/// `grid_w` x `grid_h` control points with `displacements` giving (dx, dy) at each.
/// Grid points are evenly spaced across the image. For each pixel, find the
/// enclosing grid cell, bilinearly interpolate displacement, and sample.
pub fn mesh_warp(
    input: &[f32],
    w: u32,
    h: u32,
    grid_w: u32,
    grid_h: u32,
    displacements: &[(f32, f32)],
) -> Vec<f32> {
    assert_eq!(
        displacements.len(),
        (grid_w * grid_h) as usize,
        "displacements must have grid_w * grid_h entries"
    );

    let len = (w * h * 4) as usize;
    let mut output = vec![0.0f32; len];

    // Grid cell size in pixels
    let cell_w = if grid_w > 1 {
        (w as f32 - 1.0) / (grid_w as f32 - 1.0)
    } else {
        w as f32
    };
    let cell_h = if grid_h > 1 {
        (h as f32 - 1.0) / (grid_h as f32 - 1.0)
    } else {
        h as f32
    };

    for y in 0..h {
        for x in 0..w {
            // Find grid cell
            let gx = x as f32 / cell_w;
            let gy = y as f32 / cell_h;
            let gx0 = (gx.floor() as u32).min(grid_w.saturating_sub(2));
            let gy0 = (gy.floor() as u32).min(grid_h.saturating_sub(2));
            let gx1 = (gx0 + 1).min(grid_w - 1);
            let gy1 = (gy0 + 1).min(grid_h - 1);

            let fx = (gx - gx0 as f32).clamp(0.0, 1.0);
            let fy = (gy - gy0 as f32).clamp(0.0, 1.0);

            let d00 = displacements[(gy0 * grid_w + gx0) as usize];
            let d10 = displacements[(gy0 * grid_w + gx1) as usize];
            let d01 = displacements[(gy1 * grid_w + gx0) as usize];
            let d11 = displacements[(gy1 * grid_w + gx1) as usize];

            let disp_x = d00.0 * (1.0 - fx) * (1.0 - fy)
                + d10.0 * fx * (1.0 - fy)
                + d01.0 * (1.0 - fx) * fy
                + d11.0 * fx * fy;
            let disp_y = d00.1 * (1.0 - fx) * (1.0 - fy)
                + d10.1 * fx * (1.0 - fy)
                + d01.1 * (1.0 - fx) * fy
                + d11.1 * fx * fy;

            let sx = x as f32 + disp_x;
            let sy = y as f32 + disp_y;
            let px = bilinear_sample(input, w, h, sx, sy);
            let idx = (y * w + x) as usize * 4;
            output[idx..idx + 4].copy_from_slice(&px);
        }
    }
    output
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 32;
    const H: u32 = 32;
    const TOL: f32 = 1e-5;

    fn test_image() -> Vec<f32> {
        crate::gradient(W, H)
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn barrel_identity() {
        let img = test_image();
        let out = barrel(&img, W, H, 0.0, 0.0);
        assert!(
            max_diff(&img, &out) < TOL,
            "barrel with k1=0, k2=0 should be identity"
        );
    }

    #[test]
    fn swirl_identity() {
        let img = test_image();
        let out = swirl(&img, W, H, 0.0, 100.0);
        assert!(
            max_diff(&img, &out) < TOL,
            "swirl with angle=0 should be identity"
        );
    }

    #[test]
    fn wave_identity() {
        let img = test_image();
        let out_h = wave(&img, W, H, 0.0, 10.0, true);
        let out_v = wave(&img, W, H, 0.0, 10.0, false);
        assert!(
            max_diff(&img, &out_h) < TOL,
            "wave horizontal with amplitude=0 should be identity"
        );
        assert!(
            max_diff(&img, &out_v) < TOL,
            "wave vertical with amplitude=0 should be identity"
        );
    }

    #[test]
    fn polar_depolar_roundtrip() {
        let img = test_image();
        let polar_img = polar(&img, W, H);
        let roundtrip = depolar(&polar_img, W, H);
        // Roundtrip has inherent sampling loss — use generous tolerance.
        // Check only interior pixels (edges lose data due to clamping).
        let mut max_err = 0.0f32;
        for y in 4..(H - 4) {
            for x in 4..(W - 4) {
                let idx = (y * W + x) as usize * 4;
                for c in 0..4 {
                    let err = (img[idx + c] - roundtrip[idx + c]).abs();
                    max_err = max_err.max(err);
                }
            }
        }
        assert!(
            max_err < 0.15,
            "polar->depolar roundtrip interior max error {max_err} exceeds tolerance"
        );
    }

    #[test]
    fn liquify_zero_radius_identity() {
        let img = test_image();
        let out = liquify(&img, W, H, 16.0, 16.0, 0.0, 10.0, 10.0);
        assert!(
            max_diff(&img, &out) < TOL,
            "liquify with radius=0 should be identity"
        );
    }

    #[test]
    fn barrel_produces_different_output() {
        let img = test_image();
        let out = barrel(&img, W, H, 0.5, 0.1);
        assert!(
            max_diff(&img, &out) > 0.01,
            "barrel with non-zero k should differ from input"
        );
    }

    #[test]
    fn displacement_map_neutral_identity() {
        // A map with all channels at 0.5 should produce identity
        let img = test_image();
        let neutral_map = crate::solid(W, H, [0.5, 0.5, 0.5, 1.0]);
        let out = displacement_map(&img, W, H, &neutral_map, 10.0, 10.0);
        assert!(
            max_diff(&img, &out) < TOL,
            "displacement map with neutral (0.5) map should be identity"
        );
    }

    #[test]
    fn mesh_warp_zero_displacement_identity() {
        let img = test_image();
        let grid_w = 4u32;
        let grid_h = 4u32;
        let displacements = vec![(0.0f32, 0.0f32); (grid_w * grid_h) as usize];
        let out = mesh_warp(&img, W, H, grid_w, grid_h, &displacements);
        assert!(
            max_diff(&img, &out) < TOL,
            "mesh warp with zero displacements should be identity"
        );
    }
}
