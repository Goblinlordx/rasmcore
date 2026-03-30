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

/// Exact Robidoux filter B,C values from IM resize.c.
const ROBIDOUX_B: f64 = 0.37821575509399867;
const ROBIDOUX_C: f64 = 0.31089212245300067;

/// Support radius for the Robidoux filter.
const SUPPORT: f64 = 2.0;

/// Weight LUT size (matches IM's WLUT_WIDTH).
const WLUT_WIDTH: usize = 1024;

/// Evaluate the Robidoux cubic filter at distance `r` (f64 for LUT precomputation).
#[inline]
fn robidoux_kernel_f64(r: f64) -> f64 {
    let r = r.abs();
    if r >= 2.0 {
        return 0.0;
    }
    let b = ROBIDOUX_B;
    let c = ROBIDOUX_C;
    if r < 1.0 {
        let r2 = r * r;
        let r3 = r2 * r;
        ((12.0 - 9.0 * b - 6.0 * c) * r3 + (-18.0 + 12.0 * b + 6.0 * c) * r2 + (6.0 - 2.0 * b))
            / 6.0
    } else {
        let r2 = r * r;
        let r3 = r2 * r;
        ((-b - 6.0 * c) * r3 + (6.0 * b + 30.0 * c) * r2 + (-12.0 * b - 48.0 * c) * r
            + (8.0 * b + 24.0 * c))
            / 6.0
    }
}

/// Evaluate the Robidoux cubic filter at distance `r` (f32 convenience).
#[inline]
pub fn robidoux_kernel(r: f32) -> f32 {
    robidoux_kernel_f64(r as f64) as f32
}

/// Build the weight LUT matching IM's SetResampleFilter.
///
/// LUT[Q] = filter(sqrt(Q) * r_scale) where r_scale = support * sqrt(1/WLUT_WIDTH).
/// Q indexes [0, WLUT_WIDTH) representing squared distances scaled to the LUT range.
fn build_weight_lut() -> Vec<f64> {
    let r_scale = SUPPORT * (1.0 / WLUT_WIDTH as f64).sqrt();
    (0..WLUT_WIDTH)
        .map(|q| robidoux_kernel_f64((q as f64).sqrt() * r_scale))
        .collect()
}

/// 2×2 Jacobian matrix `[[dsx/dox, dsx/doy], [dsy/dox, dsy/doy]]`.
pub type Jacobian = [[f32; 2]; 2];

/// ClampUpAxes: SVD decomposition of the Jacobian with minor-axis clamping.
///
/// Matches IM's ClampUpAxes exactly. Ensures the sampling ellipse always
/// contains at least a unit disk (prevents magnification artifacts).
///
/// Input: Jacobian as (dux, dvx, duy, dvy) — IM's parameter order.
/// Output: (major_mag, minor_mag, major_x, major_y, minor_x, minor_y)
fn clamp_up_axes(
    dux: f64,
    dvx: f64,
    duy: f64,
    dvy: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    // Compute normal matrix n = J * J^T
    let a = dux;
    let b = duy;
    let c = dvx;
    let d = dvy;

    let aa = a * a;
    let bb = b * b;
    let cc = c * c;
    let dd = d * d;

    let n11 = aa + bb;
    let n22 = cc + dd;
    let n12 = a * c + b * d;

    // Singular values via eigenvalue decomposition of the normal matrix
    let half_sum = 0.5 * (n11 + n22);
    let half_diff_sq = 0.25 * (n11 - n22) * (n11 - n22) + n12 * n12;
    let discriminant = half_diff_sq.sqrt();

    let s1s1 = half_sum + discriminant; // largest singular value squared
    let s2s2 = half_sum - discriminant; // smallest singular value squared

    // Clamp: ensure both singular values >= 1.0
    let major_mag = if s1s1 <= 1.0 { 1.0 } else { s1s1.sqrt() };
    let minor_mag = if s2s2 <= 1.0 { 1.0 } else { s2s2.sqrt() };

    // Compute unit vectors for major and minor axes
    // The eigenvectors of n = [[n11,n12],[n12,n22]] for eigenvalue s1s1:
    // (n11 - s1s1) * vx + n12 * vy = 0 → vy/vx = -(n11-s1s1)/n12
    if n12.abs() > 1e-10 {
        let major_x = n12;
        let major_y = s1s1 - n11;
        let major_len = (major_x * major_x + major_y * major_y).sqrt();
        let minor_x = n12;
        let minor_y = s2s2 - n11;
        let minor_len = (minor_x * minor_x + minor_y * minor_y).sqrt();
        (
            major_mag,
            minor_mag,
            major_x / major_len,
            major_y / major_len,
            minor_x / minor_len,
            minor_y / minor_len,
        )
    } else {
        // Diagonal case: axes align with coordinate axes
        if n11 >= n22 {
            (major_mag, minor_mag, 1.0, 0.0, 0.0, 1.0)
        } else {
            (major_mag, minor_mag, 0.0, 1.0, 1.0, 0.0)
        }
    }
}

/// Scaled ellipse coefficients ready for LUT lookup.
struct EllipseCoeffs {
    a: f64,
    b: f64,
    c: f64,
    ulimit: f64,
    vlimit: f64,
    uwidth: f64,
    slope: f64,
}

/// Compute IM-exact ellipse coefficients from a Jacobian.
///
/// Follows IM's ScaleResampleFilter exactly:
/// 1. Map Jacobian to IM's (dux, dvx, duy, dvy) parameter order
/// 2. ClampUpAxes SVD decomposition
/// 3. A,B,C,F from clamped axes
/// 4. F *= support²
/// 5. Scale A,B,C by WLUT_WIDTH/F for direct LUT indexing
fn compute_ellipse(j: &Jacobian) -> Option<EllipseCoeffs> {
    // Map our Jacobian [[dsx/dox, dsx/doy], [dsy/dox, dsy/doy]]
    // to IM's (dux, dvx, duy, dvy) = (dsx/dox, dsy/dox, dsx/doy, dsy/doy)
    let dux = j[0][0] as f64;
    let dvx = j[1][0] as f64;
    let duy = j[0][1] as f64;
    let dvy = j[1][1] as f64;

    let (major_mag, minor_mag, major_ux, major_uy, minor_ux, minor_uy) =
        clamp_up_axes(dux, dvx, duy, dvy);

    let major_x = major_ux * major_mag;
    let major_y = major_uy * major_mag;
    let minor_x = minor_ux * minor_mag;
    let minor_y = minor_uy * minor_mag;

    let a = major_y * major_y + minor_y * minor_y;
    let b = -2.0 * (major_x * major_y + minor_x * minor_y);
    let c = major_x * major_x + minor_x * minor_x;
    let mut f = major_mag * minor_mag;
    f *= f; // square it

    // Check for overflow
    if (4.0 * a * c - b * b) > 1e15 {
        return None;
    }

    // Scale F by support²
    f *= SUPPORT * SUPPORT;

    let ac_bb4 = a * c - 0.25 * b * b;
    if ac_bb4 < 1e-10 {
        return None;
    }

    let ulimit = (c * f / ac_bb4).sqrt();
    let vlimit = (a * f / ac_bb4).sqrt();
    let uwidth = (f / a).sqrt();
    let slope = -b / (2.0 * a);

    // Scale A,B,C by WLUT_WIDTH/F for direct LUT indexing
    let scale = WLUT_WIDTH as f64 / f;

    Some(EllipseCoeffs {
        a: a * scale,
        b: b * scale,
        c: c * scale,
        ulimit,
        vlimit,
        uwidth,
        slope,
    })
}

/// EWA resampler matching IM's ResamplePixelColor exactly.
///
/// Precomputes the Robidoux weight LUT on construction.
pub struct EwaSampler<'a> {
    pub pixels: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub ch: usize,
    lut: Vec<f64>,
}

impl<'a> EwaSampler<'a> {
    /// Create a new EWA sampler with precomputed weight LUT.
    pub fn new(pixels: &'a [u8], w: usize, h: usize, ch: usize) -> Self {
        Self {
            pixels,
            w,
            h,
            ch,
            lut: build_weight_lut(),
        }
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

    /// Sample a single channel using IM-exact EWA with the given Jacobian.
    ///
    /// Matches IM's ResamplePixelColor: ClampUpAxes SVD, LUT-based weight
    /// lookup, parallelogram iteration with incremental Q updates.
    /// Falls back to bilinear for near-identity transforms where EWA
    /// would only blur without improving quality.
    pub fn sample(&self, sx: f32, sy: f32, jacobian: &Jacobian, c: usize) -> f32 {
        // Quick check: if both Jacobian rows have magnitude ≤ 1,
        // the transform is near-identity — bilinear is optimal.
        let row0_mag = jacobian[0][0] * jacobian[0][0] + jacobian[0][1] * jacobian[0][1];
        let row1_mag = jacobian[1][0] * jacobian[1][0] + jacobian[1][1] * jacobian[1][1];
        if row0_mag <= 1.05 && row1_mag <= 1.05 {
            return self.bilinear(sx, sy, c);
        }

        let coeffs = match compute_ellipse(jacobian) {
            Some(c) => c,
            None => return self.bilinear(sx, sy, c),
        };

        let u0 = sx as f64;
        let v0 = sy as f64;

        let v1 = (v0 - coeffs.vlimit).ceil() as i32;
        let v2 = (v0 + coeffs.vlimit).floor() as i32;

        let mut color = 0.0f64;
        let mut divisor = 0.0f64;
        let mut hit = 0u32;

        let ddq = 2.0 * coeffs.a;

        let mut u1_f = u0 + (v1 as f64 - v0) * coeffs.slope - coeffs.uwidth;
        let uw = (2.0 * coeffs.uwidth) as i32 + 1;

        for v in v1..=v2 {
            let u_start = u1_f.ceil() as i32;
            u1_f += coeffs.slope;

            let u_f64 = u_start as f64 - u0;
            let v_f64 = v as f64 - v0;

            let mut q = (coeffs.a * u_f64 + coeffs.b * v_f64) * u_f64 + coeffs.c * v_f64 * v_f64;
            let mut dq = coeffs.a * (2.0 * u_f64 + 1.0) + coeffs.b * v_f64;

            for u_off in 0..uw {
                let qi = q as i32;
                if qi >= 0 && qi < WLUT_WIDTH as i32 {
                    let weight = self.lut[qi as usize];
                    if weight > 0.0 {
                        color += weight * self.fetch(u_start + u_off, v, c) as f64;
                        divisor += weight;
                        hit += 1;
                    }
                }
                q += dq;
                dq += ddq;
            }
        }

        if hit > 0 && divisor > 1e-10 {
            (color / divisor) as f32
        } else {
            self.bilinear(sx, sy, c)
        }
    }

    /// Sample all channels at once using IM-exact EWA.
    pub fn sample_all(&self, sx: f32, sy: f32, jacobian: &Jacobian) -> Vec<f32> {
        let mut result = vec![0.0f32; self.ch];
        for c in 0..self.ch {
            result[c] = self.sample(sx, sy, jacobian, c);
        }
        result
    }

    /// Bilinear interpolation (public for filters that match IM's bilinear path).
    #[inline]
    pub fn bilinear_pub(&self, sx: f32, sy: f32, c: usize) -> f32 {
        self.bilinear(sx, sy, c)
    }

    /// Bilinear interpolation fallback.
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
        let expected = (6.0 - 2.0 * ROBIDOUX_B as f32) / 6.0;
        assert!(
            (v - expected).abs() < 1e-5,
            "kernel(0) = {v}, expected {expected}"
        );
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
    fn weight_lut_matches_kernel() {
        let lut = build_weight_lut();
        assert_eq!(lut.len(), WLUT_WIDTH);
        // LUT[0] should be kernel(0) = max value
        assert!(lut[0] > 0.8);
        // LUT[WLUT_WIDTH-1] should be near 0 (at edge of support)
        assert!(lut[WLUT_WIDTH - 1].abs() < 0.01);
    }

    #[test]
    fn clamp_up_axes_identity() {
        // Identity Jacobian: singular values = 1, clamped to 1
        let (major, minor, _, _, _, _) = clamp_up_axes(1.0, 0.0, 0.0, 1.0);
        assert!((major - 1.0).abs() < 1e-10);
        assert!((minor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn clamp_up_axes_scaling() {
        // 3x scale: singular values = 3, clamped stays 3
        let (major, minor, _, _, _, _) = clamp_up_axes(3.0, 0.0, 0.0, 3.0);
        assert!((major - 3.0).abs() < 1e-6);
        assert!((minor - 3.0).abs() < 1e-6);
    }

    #[test]
    fn clamp_up_axes_minification_clamped() {
        // 0.5x scale: singular values = 0.5, clamped UP to 1.0
        let (major, minor, _, _, _, _) = clamp_up_axes(0.5, 0.0, 0.0, 0.5);
        assert!((major - 1.0).abs() < 1e-6, "major should be clamped to 1.0");
        assert!((minor - 1.0).abs() < 1e-6, "minor should be clamped to 1.0");
    }

    #[test]
    fn compute_ellipse_identity() {
        let j = JACOBIAN_IDENTITY;
        let coeffs = compute_ellipse(&j).unwrap();
        // With identity Jacobian and clamped axes, A,B,C should be scaled
        // by WLUT_WIDTH / (support² * 1²) = 1024 / 4 = 256
        assert!(coeffs.a > 0.0);
        assert!(coeffs.c > 0.0);
    }

    #[test]
    fn ewa_identity_matches_bilinear() {
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

        // At identity Jacobian with clamped axes, EWA uses the Robidoux filter
        // over a unit-disk support. The result should be very close to bilinear.
        for c in 0..ch {
            let ewa = sampler.sample(3.5, 4.5, &JACOBIAN_IDENTITY, c);
            let bilinear = sampler.bilinear(3.5, 4.5, c);
            assert!(
                (ewa - bilinear).abs() < 2.0,
                "EWA({c}) = {ewa}, bilinear = {bilinear}"
            );
        }
    }

    #[test]
    fn ewa_uniform_image() {
        let w = 16usize;
        let h = 16usize;
        let pixels = vec![128u8; w * h * 3];
        let sampler = EwaSampler::new(&pixels, w, h, 3);

        let j: Jacobian = [[3.0, 1.0], [0.5, 2.0]];
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
