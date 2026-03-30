//! Elliptical Weighted Average (EWA) resampling engine.
//!
//! Provides high-quality area resampling for distortion filters, matching
//! ImageMagick's `-distort` quality. Uses a Robidoux filter kernel with
//! Jacobian-based elliptical support region.
//!
//! # Algorithm (Heckbert 1989, as implemented by IM)
//!
//! For each output pixel:
//! 1. Compute source coordinates `(sx, sy)` via inverse distortion
//! 2. Compute 2×2 Jacobian matrix of the transformation
//! 3. From the Jacobian, derive an ellipse in source space
//! 4. Iterate source pixels within the ellipse, weight by Robidoux kernel
//! 5. Accumulate weighted colors; divide by total weight
//!
//! # Robidoux filter
//!
//! Cylindrical cubic with B=0.3782, C=0.3109 (Nicolas Robidoux optimal values).
//! Support radius = 2.0. Matches IM's default for `-distort` operations.

/// Robidoux filter coefficients (Mitchell-Netravali cubic with B=0.3782, C=0.3109).
const ROBIDOUX_B: f32 = 12.0 / (19.0 + 9.0 * core::f32::consts::FRAC_1_SQRT_2);
const ROBIDOUX_C: f32 = 113.0 / (58.0 + 216.0 * core::f32::consts::FRAC_1_SQRT_2);

/// Maximum support radius for the Robidoux filter.
const SUPPORT: f32 = 2.0;

/// Maximum number of source pixels to sample (prevents runaway for extreme distortions).
const MAX_SAMPLES: usize = 4096;

/// Evaluate the Robidoux cubic filter at distance `r`.
///
/// Returns the filter weight for the given radial distance.
/// Zero outside the support radius (r > 2).
#[inline]
pub fn robidoux_kernel(r: f32) -> f32 {
    let r = r.abs();
    if r >= 2.0 {
        return 0.0;
    }
    let b = ROBIDOUX_B;
    let c = ROBIDOUX_C;
    if r < 1.0 {
        // Inner lobe: ((12 - 9B - 6C)*r³ + (-18 + 12B + 6C)*r² + (6 - 2B)) / 6
        let r2 = r * r;
        let r3 = r2 * r;
        ((12.0 - 9.0 * b - 6.0 * c) * r3 + (-18.0 + 12.0 * b + 6.0 * c) * r2 + (6.0 - 2.0 * b))
            / 6.0
    } else {
        // Outer lobe: ((-B - 6C)*r³ + (6B + 30C)*r² + (-12B - 48C)*r + (8B + 24C)) / 6
        let r2 = r * r;
        let r3 = r2 * r;
        ((-b - 6.0 * c) * r3 + (6.0 * b + 30.0 * c) * r2 + (-12.0 * b - 48.0 * c) * r
            + (8.0 * b + 24.0 * c))
            / 6.0
    }
}

/// 2×2 Jacobian matrix `[[dsx/dox, dsx/doy], [dsy/dox, dsy/doy]]`.
pub type Jacobian = [[f32; 2]; 2];

/// Compute EWA ellipse parameters from a Jacobian matrix.
///
/// Returns `(a, b, c, f)` where the ellipse equation is:
/// `Q = a*(x-sx)² + 2*b*(x-sx)*(y-sy) + c*(y-sy)²`
/// and `f = a*c - b²` is the normalization factor.
///
/// The ellipse semi-axes are derived from the eigenvalues of J^T·J.
#[inline]
fn ellipse_from_jacobian(j: &Jacobian) -> (f32, f32, f32, f32) {
    // J = [[j00, j01], [j10, j11]]
    // Ellipse coefficients from J^T * J:
    //   A = j00² + j10²
    //   B = j00*j01 + j10*j11
    //   C = j01² + j11²
    let a = j[0][0] * j[0][0] + j[1][0] * j[1][0];
    let b = j[0][0] * j[0][1] + j[1][0] * j[1][1];
    let c = j[0][1] * j[0][1] + j[1][1] * j[1][1];
    let f = a * c - b * b; // determinant (area scale factor²)

    // Clamp F to avoid division by zero for degenerate transforms
    let f = f.max(1e-10);

    (a, b, c, f)
}

/// EWA resampler for distortion filters.
///
/// Holds a reference to the source image and provides `sample()` for
/// Robidoux-filtered elliptical sampling.
pub struct EwaSampler<'a> {
    pub pixels: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub ch: usize,
}

impl<'a> EwaSampler<'a> {
    /// Create a new EWA sampler for the given image.
    pub fn new(pixels: &'a [u8], w: usize, h: usize, ch: usize) -> Self {
        Self { pixels, w, h, ch }
    }

    /// Fetch a single channel value at integer coordinates with bounds check.
    #[inline]
    fn fetch(&self, px: i32, py: i32, c: usize) -> f32 {
        if px >= 0 && (px as usize) < self.w && py >= 0 && (py as usize) < self.h {
            self.pixels[(py as usize * self.w + px as usize) * self.ch + c] as f32
        } else {
            0.0
        }
    }

    /// Sample a single channel using EWA with the given Jacobian.
    ///
    /// `sx, sy`: source coordinates (floating point)
    /// `jacobian`: 2×2 Jacobian of the distortion at this pixel
    /// `c`: channel index
    ///
    /// Falls back to bilinear for identity/near-identity Jacobians (optimization).
    pub fn sample(&self, sx: f32, sy: f32, jacobian: &Jacobian, c: usize) -> f32 {
        let (a, b, cc, f) = ellipse_from_jacobian(jacobian);

        // For near-identity transforms (unit Jacobian), the ellipse collapses
        // to a point and EWA degenerates to bilinear. Use bilinear directly
        // when the ellipse is smaller than ~1 pixel in both axes.
        let trace = a + cc;
        if trace < 2.5 {
            return self.bilinear(sx, sy, c);
        }

        // Compute bounding box of the ellipse in source space.
        // Semi-axes of the ellipse: eigenvalues of the quadratic form.
        // The bounding box is ±sqrt(C/F)*support for x, ±sqrt(A/F)*support for y.
        let inv_f = 1.0 / f;
        let ux = (cc * inv_f).sqrt() * SUPPORT;
        let uy = (a * inv_f).sqrt() * SUPPORT;

        // Clamp bounding box to reasonable size
        let ux = ux.min(self.w as f32);
        let uy = uy.min(self.h as f32);

        let x_min = (sx - ux).floor() as i32;
        let x_max = (sx + ux).ceil() as i32;
        let y_min = (sy - uy).floor() as i32;
        let y_max = (sy + uy).ceil() as i32;

        // Accumulate weighted samples
        let mut color = 0.0f32;
        let mut weight_sum = 0.0f32;
        let mut sample_count = 0usize;

        // Precompute normalization: we evaluate Q/F and pass sqrt(Q/F) to the kernel
        let a_norm = a * inv_f;
        let b_norm = b * inv_f;
        let c_norm = cc * inv_f;

        for iy in y_min..=y_max {
            let dy = iy as f32 - sy;
            // Partial quadratic: a_norm * dy² (for the y component)
            let q_y = c_norm * dy * dy;
            let q_xy_base = 2.0 * b_norm * dy;

            for ix in x_min..=x_max {
                let dx = ix as f32 - sx;

                // Full quadratic: Q/F = a_norm*dx² + 2*b_norm*dx*dy + c_norm*dy²
                let q = a_norm * dx * dx + q_xy_base * dx + q_y;

                // q is the squared normalized distance. Filter support is r < SUPPORT,
                // so Q/F < SUPPORT² = 4.0
                if q >= SUPPORT * SUPPORT {
                    continue;
                }

                let r = q.sqrt();
                let w = robidoux_kernel(r);
                if w == 0.0 {
                    continue;
                }

                color += self.fetch(ix, iy, c) * w;
                weight_sum += w;

                sample_count += 1;
                if sample_count >= MAX_SAMPLES {
                    break;
                }
            }
            if sample_count >= MAX_SAMPLES {
                break;
            }
        }

        if weight_sum > 0.0 {
            color / weight_sum
        } else {
            self.bilinear(sx, sy, c)
        }
    }

    /// Sample all channels at once using EWA.
    ///
    /// Returns a Vec of channel values (length = self.ch).
    pub fn sample_all(&self, sx: f32, sy: f32, jacobian: &Jacobian) -> Vec<f32> {
        let (a, b, cc, f) = ellipse_from_jacobian(jacobian);

        let trace = a + cc;
        if trace < 2.5 {
            let mut result = vec![0.0f32; self.ch];
            for c in 0..self.ch {
                result[c] = self.bilinear(sx, sy, c);
            }
            return result;
        }

        let inv_f = 1.0 / f;
        let ux = (cc * inv_f).sqrt() * SUPPORT;
        let uy = (a * inv_f).sqrt() * SUPPORT;
        let ux = ux.min(self.w as f32);
        let uy = uy.min(self.h as f32);

        let x_min = (sx - ux).floor() as i32;
        let x_max = (sx + ux).ceil() as i32;
        let y_min = (sy - uy).floor() as i32;
        let y_max = (sy + uy).ceil() as i32;

        let mut colors = vec![0.0f32; self.ch];
        let mut weight_sum = 0.0f32;
        let mut sample_count = 0usize;

        let a_norm = a * inv_f;
        let b_norm = b * inv_f;
        let c_norm = cc * inv_f;

        for iy in y_min..=y_max {
            let dy = iy as f32 - sy;
            let q_y = c_norm * dy * dy;
            let q_xy_base = 2.0 * b_norm * dy;

            for ix in x_min..=x_max {
                let dx = ix as f32 - sx;
                let q = a_norm * dx * dx + q_xy_base * dx + q_y;

                if q >= SUPPORT * SUPPORT {
                    continue;
                }

                let r = q.sqrt();
                let w = robidoux_kernel(r);
                if w == 0.0 {
                    continue;
                }

                // Fetch all channels for this source pixel
                if ix >= 0
                    && (ix as usize) < self.w
                    && iy >= 0
                    && (iy as usize) < self.h
                {
                    let base = (iy as usize * self.w + ix as usize) * self.ch;
                    for c in 0..self.ch {
                        colors[c] += self.pixels[base + c] as f32 * w;
                    }
                }
                // Out-of-bounds contributes 0 * weight (black border)
                weight_sum += w;

                sample_count += 1;
                if sample_count >= MAX_SAMPLES {
                    break;
                }
            }
            if sample_count >= MAX_SAMPLES {
                break;
            }
        }

        if weight_sum > 0.0 {
            for c in &mut colors {
                *c /= weight_sum;
            }
        } else {
            for c in 0..self.ch {
                colors[c] = self.bilinear(sx, sy, c);
            }
        }

        colors
    }

    /// Bilinear interpolation fallback (for near-identity transforms).
    #[inline]
    fn bilinear(&self, sx: f32, sy: f32, c: usize) -> f32 {
        let x0 = sx.floor() as i32;
        let y0 = sy.floor() as i32;
        let fx = sx - x0 as f32;
        let fy = sy - y0 as f32;
        self.fetch(x0, y0, c) * (1.0 - fx) * (1.0 - fy)
            + self.fetch(x0 + 1, y0, c) * fx * (1.0 - fy)
            + self.fetch(x0, y0 + 1, c) * (1.0 - fx) * fy
            + self.fetch(x0 + 1, y0 + 1, c) * fx * fy
    }
}

// ─── Jacobian helpers for each distortion type ───────────────────────────────

/// Identity Jacobian (no distortion). EWA degrades to bilinear.
pub const JACOBIAN_IDENTITY: Jacobian = [[1.0, 0.0], [0.0, 1.0]];

/// Compute Jacobian for polar (Cartesian→Polar) distortion.
///
/// Forward: `angle = x/w*2π - π; radius = y/h * max_r`
///          `sx = cx + radius * sin(angle); sy = cy - radius * cos(angle)`
///
/// Derivatives:
///   `dsx/dox = (max_r/h) * sin(angle) * 0  +  radius * cos(angle) * (2π/w)`
///   ... simplified via chain rule
#[inline]
pub fn jacobian_polar(
    ox: f32,
    oy: f32,
    w: f32,
    h: f32,
    max_r: f32,
) -> Jacobian {
    let two_pi = std::f32::consts::TAU;
    let angle = ox / w * two_pi - std::f32::consts::PI;
    let radius = oy / h * max_r;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let da_dx = two_pi / w;
    let dr_dy = max_r / h;

    // sx = cx + radius * sin(angle)
    // dsx/dox = radius * cos(angle) * da_dx
    // dsx/doy = sin(angle) * dr_dy
    // sy = cy - radius * cos(angle)
    // dsy/dox = radius * sin(angle) * da_dx
    // dsy/doy = -cos(angle) * dr_dy
    [
        [radius * cos_a * da_dx, sin_a * dr_dy],
        [radius * sin_a * da_dx, -cos_a * dr_dy],
    ]
}

/// Compute Jacobian for depolar (IM Polar) distortion.
///
/// Forward: `ii = ox - cx; jj = oy - cy; angle = atan2(ii, jj)`
///          `sx = angle/(2π) * w + w/2; sy = hypot(ii, jj) * (h/max_r)`
#[inline]
pub fn jacobian_depolar(
    ox: f32,
    oy: f32,
    cx: f32,
    cy: f32,
    c6: f32,  // w/(2π)
    c7: f32,  // h/max_r
) -> Jacobian {
    let ii = ox - cx;
    let jj = oy - cy;
    let r2 = ii * ii + jj * jj;
    if r2 < 1e-10 {
        return JACOBIAN_IDENTITY;
    }
    let r = r2.sqrt();

    // sx = atan2(ii, jj) * c6 + w/2
    // d(atan2(ii, jj))/dii = jj / r²
    // d(atan2(ii, jj))/djj = -ii / r²
    let dsx_dox = jj / r2 * c6;
    let dsx_doy = -ii / r2 * c6;

    // sy = r * c7
    // dr/dii = ii / r, dr/djj = jj / r
    let dsy_dox = ii / r * c7;
    let dsy_doy = jj / r * c7;

    [[dsx_dox, dsx_doy], [dsy_dox, dsy_doy]]
}

/// Compute Jacobian for wave distortion.
///
/// Horizontal wave: `sx = ox; sy = oy - amp * sin(2π * ox / wl)`
/// Vertical wave: `sx = ox - amp * sin(2π * oy / wl); sy = oy`
#[inline]
pub fn jacobian_wave(
    ox: f32,
    _oy: f32,
    amplitude: f32,
    wavelength: f32,
    is_vertical: bool,
) -> Jacobian {
    let two_pi = std::f32::consts::TAU;
    let k = two_pi / wavelength;
    if is_vertical {
        // sx = ox - amp * sin(k * oy)
        // dsx/dox = 1, dsx/doy = -amp * k * cos(k * oy)
        let dsx_doy = -amplitude * k * (k * _oy).cos();
        [[1.0, dsx_doy], [0.0, 1.0]]
    } else {
        // sy = oy - amp * sin(k * ox)
        // dsy/dox = -amp * k * cos(k * ox), dsy/doy = 1
        let dsy_dox = -amplitude * k * (k * ox).cos();
        [[1.0, 0.0], [dsy_dox, 1.0]]
    }
}

/// Compute Jacobian for ripple distortion.
///
/// `disp = amp * sin(2π * r / wl); sx = ox + disp * cos(θ); sy = oy + disp * sin(θ)`
/// where `r = hypot(ox-cx, oy-cy)`, `θ = atan2(oy-cy, ox-cx)`
#[inline]
pub fn jacobian_ripple(
    ox: f32,
    oy: f32,
    cx: f32,
    cy: f32,
    amplitude: f32,
    wavelength: f32,
) -> Jacobian {
    let dx = ox - cx;
    let dy = oy - cy;
    let r = (dx * dx + dy * dy).sqrt();
    if r < 1e-6 {
        return JACOBIAN_IDENTITY;
    }
    let two_pi = std::f32::consts::TAU;
    let k = two_pi / wavelength;
    let cos_a = dx / r;
    let sin_a = dy / r;
    let sin_kr = (k * r).sin();
    let cos_kr = (k * r).cos();

    // disp = amp * sin(k * r)
    // d(disp)/dx = amp * k * cos(k*r) * (dx/r)
    // d(disp)/dy = amp * k * cos(k*r) * (dy/r)
    let dd_dx = amplitude * k * cos_kr * cos_a;
    let dd_dy = amplitude * k * cos_kr * sin_a;

    // cos_a = dx/r: d(cos_a)/dx = (1 - cos_a²)/r = sin_a²/r
    // d(cos_a)/dy = -cos_a*sin_a/r
    // sin_a = dy/r: d(sin_a)/dx = -cos_a*sin_a/r
    // d(sin_a)/dy = cos_a²/r

    let disp = amplitude * sin_kr;
    let inv_r = 1.0 / r;

    // sx = ox + disp * cos_a
    let dsx_dox = 1.0 + dd_dx * cos_a + disp * sin_a * sin_a * inv_r;
    let dsx_doy = dd_dy * cos_a + disp * (-cos_a * sin_a * inv_r);

    // sy = oy + disp * sin_a
    let dsy_dox = dd_dx * sin_a + disp * (-cos_a * sin_a * inv_r);
    let dsy_doy = 1.0 + dd_dy * sin_a + disp * cos_a * cos_a * inv_r;

    [[dsx_dox, dsx_doy], [dsy_dox, dsy_doy]]
}

/// Compute Jacobian for swirl distortion.
///
/// `rot = angle_rad * max(1 - r/radius, 0); sx = cos(rot)*dx - sin(rot)*dy + cx`
#[inline]
pub fn jacobian_swirl(
    ox: f32,
    oy: f32,
    cx: f32,
    cy: f32,
    angle_rad: f32,
    rad: f32,
    aspect_x: f32,
    aspect_y: f32,
) -> Jacobian {
    let dx = (ox - cx) * aspect_x;
    let dy = (oy - cy) * aspect_y;
    let r = (dx * dx + dy * dy).sqrt();

    if r >= rad || r < 1e-6 {
        return JACOBIAN_IDENTITY;
    }

    let t = 1.0 - r / rad;
    let t2 = t * t;
    let rot = angle_rad * t2;
    let cos_r = rot.cos();
    let sin_r = rot.sin();

    // dt/dr = -1/rad, d(t²)/dr = -2t/rad
    // drot/dr = angle_rad * (-2t/rad)
    // dr/dox = dx/r * aspect_x, dr/doy = dy/r * aspect_y
    let drot_dr = angle_rad * (-2.0 * t / rad);
    let dr_dox = if r > 0.0 { dx / r * aspect_x } else { 0.0 };
    let dr_doy = if r > 0.0 { dy / r * aspect_y } else { 0.0 };
    let drot_dox = drot_dr * dr_dox;
    let drot_doy = drot_dr * dr_doy;

    // sx = cos(rot)*dx - sin(rot)*dy + cx
    // dsx/dox = -sin(rot)*drot_dox*dx + cos(rot)*aspect_x - cos(rot)*drot_dox*dy - sin(rot)*0
    //         = cos(rot)*aspect_x + (-sin(rot)*dx - cos(rot)*dy) * drot_dox
    let dsx_dox = cos_r * aspect_x + (-sin_r * dx - cos_r * dy) * drot_dox;
    let dsx_doy = cos_r * 0.0 + (-sin_r * dx - cos_r * dy) * drot_doy - sin_r * aspect_y;

    // sy = sin(rot)*dx + cos(rot)*dy + cy
    let dsy_dox = sin_r * aspect_x + (cos_r * dx - sin_r * dy) * drot_dox;
    let dsy_doy = (cos_r * dx - sin_r * dy) * drot_doy + cos_r * aspect_y;

    [[dsx_dox, dsx_doy], [dsy_dox, dsy_doy]]
}

/// Compute Jacobian for barrel distortion.
///
/// `factor = 1 + k1*rn² + k2*rn⁴; sx = dx*factor*norm + cx`
#[inline]
pub fn jacobian_barrel(
    ox: f32,
    oy: f32,
    cx: f32,
    cy: f32,
    k1: f32,
    k2: f32,
    norm: f32,
) -> Jacobian {
    let dx = ox - cx;
    let dy = oy - cy;
    let rn = (dx * dx + dy * dy).sqrt() * norm;
    let rn2 = rn * rn;
    let rn4 = rn2 * rn2;
    let factor = 1.0 + k1 * rn2 + k2 * rn4;

    // d(factor)/d(rn) = 2*k1*rn + 4*k2*rn³
    // d(rn)/d(ox) = dx * norm² / rn (if rn > 0)
    if rn < 1e-10 {
        return [[factor * norm, 0.0], [0.0, factor * norm]];
    }

    let dfactor_drn = 2.0 * k1 * rn + 4.0 * k2 * rn2 * rn;
    let drn_dox = dx * norm * norm / rn;
    let drn_doy = dy * norm * norm / rn;

    // sx = dx * factor * norm + cx
    // dsx/dox = factor*norm + dx*norm * dfactor_drn * drn_dox
    let dsx_dox = factor * norm + dx * norm * dfactor_drn * drn_dox;
    let dsx_doy = dx * norm * dfactor_drn * drn_doy;
    let dsy_dox = dy * norm * dfactor_drn * drn_dox;
    let dsy_doy = factor * norm + dy * norm * dfactor_drn * drn_doy;

    [[dsx_dox, dsx_doy], [dsy_dox, dsy_doy]]
}

/// Compute Jacobian for spherize distortion.
///
/// `new_r = r^(1/(1+amt)) for bulge, r^(1+|amt|) for pinch`
#[inline]
pub fn jacobian_spherize(
    ox: f32,
    oy: f32,
    cx: f32,
    cy: f32,
    amount: f32,
    radius: f32,
) -> Jacobian {
    let dx = (ox - cx) / radius;
    let dy = (oy - cy) / radius;
    let r = (dx * dx + dy * dy).sqrt();

    if r >= 1.0 || r < 1e-10 {
        return JACOBIAN_IDENTITY;
    }

    let exponent = if amount >= 0.0 {
        1.0 / (1.0 + amount)
    } else {
        1.0 + amount.abs()
    };
    let new_r = r.powf(exponent);
    let scale = new_r / r;

    // d(new_r)/dr = exponent * r^(exponent-1) = exponent * new_r / r
    let dnr_dr = exponent * new_r / r;
    // d(scale)/dr = (dnr_dr * r - new_r) / r² = (dnr_dr - scale) / r
    let dscale_dr = (dnr_dr - scale) / r;

    // dr/dx = dx/r, dr/dy = dy/r (in normalized coords)
    // sx = dx*scale*radius + cx = (ox-cx)*scale + cx
    // dsx/dox = scale + (ox-cx) * dscale_dr * (dx/r) / radius
    //         = scale + dx * dscale_dr * dx / r
    let dsx_dox = scale + dx * dscale_dr * dx / r;
    let dsx_doy = dx * dscale_dr * dy / r;
    let dsy_dox = dy * dscale_dr * dx / r;
    let dsy_doy = scale + dy * dscale_dr * dy / r;

    [[dsx_dox, dsx_doy], [dsy_dox, dsy_doy]]
}

/// Compute numerical Jacobian via finite differences.
///
/// For distortions without analytical derivatives (e.g., displacement_map).
/// `distort_fn` maps output (ox, oy) → source (sx, sy).
#[inline]
pub fn jacobian_numerical<F>(ox: f32, oy: f32, distort_fn: &F) -> Jacobian
where
    F: Fn(f32, f32) -> (f32, f32),
{
    let h = 0.5;
    let (sx_px, sy_px) = distort_fn(ox + h, oy);
    let (sx_mx, sy_mx) = distort_fn(ox - h, oy);
    let (sx_py, sy_py) = distort_fn(ox, oy + h);
    let (sx_my, sy_my) = distort_fn(ox, oy - h);

    let inv_2h = 1.0 / (2.0 * h);
    [
        [(sx_px - sx_mx) * inv_2h, (sx_py - sx_my) * inv_2h],
        [(sy_px - sy_mx) * inv_2h, (sy_py - sy_my) * inv_2h],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robidoux_kernel_at_zero() {
        let v = robidoux_kernel(0.0);
        // At r=0: (6 - 2B) / 6
        let expected = (6.0 - 2.0 * ROBIDOUX_B) / 6.0;
        assert!(
            (v - expected).abs() < 1e-6,
            "kernel(0) = {v}, expected {expected}"
        );
        assert!(v > 0.8, "kernel(0) should be > 0.8, got {v}");
    }

    #[test]
    fn robidoux_kernel_at_support_boundary() {
        assert_eq!(robidoux_kernel(2.0), 0.0);
        assert_eq!(robidoux_kernel(2.5), 0.0);
    }

    #[test]
    fn robidoux_kernel_continuity_at_one() {
        let below = robidoux_kernel(0.999);
        let above = robidoux_kernel(1.001);
        assert!(
            (below - above).abs() < 0.01,
            "discontinuity at r=1: {below} vs {above}"
        );
    }

    #[test]
    fn robidoux_kernel_positive_inner() {
        for i in 0..100 {
            let r = i as f32 / 100.0;
            assert!(
                robidoux_kernel(r) >= 0.0,
                "kernel({r}) = {} < 0",
                robidoux_kernel(r)
            );
        }
    }

    #[test]
    fn ellipse_identity_jacobian() {
        let j = JACOBIAN_IDENTITY;
        let (a, b, c, f) = ellipse_from_jacobian(&j);
        assert!((a - 1.0).abs() < 1e-6, "A should be 1.0, got {a}");
        assert!(b.abs() < 1e-6, "B should be 0.0, got {b}");
        assert!((c - 1.0).abs() < 1e-6, "C should be 1.0, got {c}");
        assert!((f - 1.0).abs() < 1e-6, "F should be 1.0, got {f}");
    }

    #[test]
    fn ellipse_scale_jacobian() {
        // 2x scale in both axes
        let j: Jacobian = [[2.0, 0.0], [0.0, 2.0]];
        let (a, _b, c, f) = ellipse_from_jacobian(&j);
        assert!((a - 4.0).abs() < 1e-6);
        assert!((c - 4.0).abs() < 1e-6);
        assert!((f - 16.0).abs() < 1e-6);
    }

    #[test]
    fn ewa_identity_matches_bilinear() {
        // 8×8 gradient image
        let w = 8usize;
        let h = 8usize;
        let ch = 3usize;
        let mut pixels = vec![0u8; w * h * ch];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * ch;
                pixels[i] = (x * 32) as u8;
                pixels[i + 1] = (y * 32) as u8;
                pixels[i + 2] = 128;
            }
        }

        let sampler = EwaSampler::new(&pixels, w, h, ch);

        // At identity Jacobian, EWA should match bilinear
        for c in 0..ch {
            let ewa = sampler.sample(3.5, 4.5, &JACOBIAN_IDENTITY, c);
            let bilinear = sampler.bilinear(3.5, 4.5, c);
            assert!(
                (ewa - bilinear).abs() < 0.01,
                "EWA({c}) = {ewa}, bilinear = {bilinear}"
            );
        }
    }

    #[test]
    fn ewa_uniform_image() {
        // Uniform image should produce the same value regardless of Jacobian
        let w = 16usize;
        let h = 16usize;
        let pixels = vec![128u8; w * h * 3];
        let sampler = EwaSampler::new(&pixels, w, h, 3);

        let j: Jacobian = [[3.0, 1.0], [0.5, 2.0]]; // arbitrary distortion
        for c in 0..3 {
            let val = sampler.sample(8.0, 8.0, &j, c);
            assert!(
                (val - 128.0).abs() < 1.0,
                "uniform image should give 128, got {val}"
            );
        }
    }

    #[test]
    fn jacobian_wave_identity_at_zero_amplitude() {
        let j = jacobian_wave(10.0, 20.0, 0.0, 30.0, false);
        assert!((j[0][0] - 1.0).abs() < 1e-6);
        assert!((j[1][1] - 1.0).abs() < 1e-6);
        assert!(j[0][1].abs() < 1e-6);
        assert!(j[1][0].abs() < 1e-6);
    }

    #[test]
    fn jacobian_numerical_matches_identity() {
        let j = jacobian_numerical(10.0, 20.0, &|x, y| (x, y));
        assert!((j[0][0] - 1.0).abs() < 0.01);
        assert!((j[1][1] - 1.0).abs() < 0.01);
        assert!(j[0][1].abs() < 0.01);
        assert!(j[1][0].abs() < 0.01);
    }
}
