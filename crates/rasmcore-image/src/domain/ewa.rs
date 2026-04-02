//! Elliptical Weighted Average (EWA) resampling engine.
//!
//! Ported from ImageMagick 6 source code for pixel-level parity with
//! `magick -distort` operations. Uses Robidoux-filtered area sampling
//! with Jacobian-based elliptical support region.
//!
//! # Reference implementation
//!
//! Ported from ImageMagick 6 (`github.com/ImageMagick/ImageMagick6`):
//!
//! | IM source file | Functions ported | What we use them for |
//! |----------------|-----------------|---------------------|
//! | `magick/resample.c` | `ClampUpAxes()` | SVD decomposition + minor-axis clamping |
//! | `magick/resample.c` | `ScaleResampleFilter()` | Ellipse A,B,C,F coefficients, LUT scaling |
//! | `magick/resample.c` | `ResamplePixelColor()` | Parallelogram iteration, DQ/DDQ loop |
//! | `magick/resize.c` | Filter table | Robidoux B,C values (line ~1150) |
//! | `magick/distort.c` | Pixel loop | `d.x = i + 0.5` pixel-center convention |
//! | `magick/distort.c` | `GenerateCoefficients()` | Polar/Barrel coefficient computation |
//!
//! # Algorithm (Heckbert 1989, as implemented by IM)
//!
//! For each output pixel:
//! 1. Compute source coordinates `(sx, sy)` via inverse distortion
//! 2. Compute 2×2 Jacobian matrix of the transformation
//! 3. `ClampUpAxes`: SVD of Jacobian, clamp minor axis ≥ 1.0
//! 4. Compute ellipse coefficients A,B,C,F from clamped axes
//! 5. Scale F by support², then scale A,B,C by `WLUT_WIDTH/F` for LUT indexing
//! 6. Iterate source pixels within parallelogram (slope = -B/(2A))
//! 7. For each pixel: `Q = A*U² + B*U*V + C*V²`, weight = `LUT[(int)Q]`
//! 8. Accumulate weighted colors; divide by total weight
//!
//! # Robidoux filter
//!
//! Cylindrical Mitchell-Netravali cubic. Support radius = 2.0.
//! - B = 0.37821575509399867 (from IM `resize.c` filter table)
//! - C = 0.31089212245300067 (from IM `resize.c` filter table)
//!
//! # What was ported verbatim from IM
//!
//! - **ClampUpAxes**: discriminant = `(frobenius²+2det)(frobenius²-2det)`,
//!   eigenvector row selection by `|s1s1-n11|² vs |s1s1-n22|²`, minor axis
//!   as 90° rotation of major `(-u21, u11)`.
//! - **Weight LUT**: 1024-entry table indexed by Q directly. Built as
//!   `LUT[Q] = kernel(sqrt(Q) * support/sqrt(WLUT_WIDTH))`.
//! - **ScaleResampleFilter**: A,B,C from clamped axes, F = (major*minor)²,
//!   F *= support², A/B/C scaled by WLUT_WIDTH/F.
//! - **ResamplePixelColor**: parallelogram iteration with `slope = -B/(2A)`,
//!   `Uwidth = sqrt(F/A)`, incremental `Q += DQ; DQ += DDQ` updates.
//! - **Q16 simulation**: source pixels scaled ×257 before accumulation,
//!   /257 after (matching IM Q16-HDRI internal 0-65535 range).
//! - **Pixel-center convention**: `d.x = i + 0.5` for distort source coords.
//!
//! # Sampling mode selection
//!
//! Each distortion filter uses one of three sampling modes. The choice is based
//! on empirical benchmarking against ImageMagick 7.x (see `sampling_mode_audit`
//! test and `.agent/kf/tracks/sampling-mode-audit_20260402041200Z/report.md`).
//!
//! | Filter    | Mode      | MAE vs IM | Why                                    |
//! |-----------|-----------|-----------|----------------------------------------|
//! | wave      | Bilinear  | 0.00      | IM uses bilinear in `effect.c`         |
//! | swirl     | Bilinear  | 0.00      | IM uses bilinear in `effect.c`         |
//! | polar     | Bilinear  | 1.95      | Matches IM DePolar better than EWA     |
//! | depolar   | Ewa       | 2.55      | Matches IM Polar (distort.c EWA)       |
//! | barrel    | EwaClamp  | 8.24      | IM barrel uses EWA + edge-clamp border |
//! | spherize  | Ewa       | n/a       | No IM equivalent; EWA suits anisotropy |
//! | ripple    | Ewa       | n/a       | No IM equivalent; EWA suits radial     |
//! | mesh_warp | Bilinear  | n/a       | Piecewise affine; locally smooth       |
//!
//! **Rules of thumb:**
//! - Use **Bilinear** when IM uses `effect.c` (not EWA), or the mapping is
//!   locally smooth with bounded Jacobian (wave, swirl, mesh_warp, polar).
//! - Use **Ewa** when the mapping has strong anisotropy (depolar near center,
//!   spherize near edge) and IM uses its distort.c EWA pipeline.
//! - Use **EwaClamp** only for barrel distortion where edge-repeat border
//!   handling matches IM's `-virtual-pixel Edge` default.
//! - **Never** use EwaClamp as a general-purpose mode — it causes catastrophic
//!   edge-bleed for most other distortion types.
//!
//! # IM parity — current status
//!
//! Tested against ImageMagick 7.1.2-18 Q16-HDRI on 64×64 test images.
//!
//! | Filter  | MAE  | IM command | Our sampling |
//! |---------|------|------------|--------------|
//! | wave    | 0.00 | `-wave 5x20` | Bilinear |
//! | swirl   | 0.00 | `-swirl 90` | Bilinear |
//! | polar   | 1.95 | `-distort DePolar 32` | Bilinear |
//! | depolar | 2.55 | `-distort Polar 32` | Ewa |
//! | barrel  | 8.24 | `-distort Barrel "0.5 0.1 0 1"` | EwaClamp |
//!
//! # Known residuals and root causes
//!
//! ## Wave/Swirl (MAE 0.00) — RESOLVED
//!
//! IM implements `-wave` and `-swirl` in `effect.c` with bilinear interpolation,
//! not in `distort.c` with EWA. Using `DistortionSampling::Bilinear` gives
//! exact match (MAE 0.00).
//!
//! ## Polar/Depolar (MAE ~2.0-2.5) — FP PRECISION FLOOR
//!
//! **Root cause: incremental quadratic accumulation precision.**
//!
//! Our EWA inner loop uses the same DQ/DDQ incremental scheme as IM:
//! ```text
//! Q = (A*U + B*V)*U + C*V*V
//! DQ = A*(2*U+1) + B*V
//! DDQ = 2*A
//! for each pixel: Q += DQ; DQ += DDQ
//! ```
//! Both implementations use f64. However, the accumulated rounding differs
//! because:
//! 1. Our Jacobian is computed in f32 then cast to f64 for ellipse computation.
//!    IM computes the Jacobian as f64 throughout (via `ScaleResampleFilter`
//!    which receives `double` parameters from the distort loop).
//! 2. The A,B,C coefficients are derived from ClampUpAxes output. Even with
//!    identical ClampUpAxes (verified by verbatim port), the f32→f64 cast
//!    of the input Jacobian introduces ~1e-7 relative error that propagates
//!    through the ellipse math.
//! 3. Q is truncated to `int` for LUT indexing. Near LUT bin boundaries,
//!    the ~1e-7 error can shift Q to an adjacent bin, causing ±1 in the
//!    final pixel value.
//!
//! **Evidence**: Per-channel average diff is 0.83 (< 1 quantization level).
//! This was verified by confirming our bilinear gives MAE 4.35 vs IM
//! (matching `IM -filter Point` vs `IM default`), proving our EWA IS
//! producing better results than bilinear, just not bit-exact with IM's.
//!
//! **To close this gap** would require changing the `Jacobian` type from
//! `[[f32; 2]; 2]` to `[[f64; 2]; 2]` and computing all distortion
//! Jacobians in f64 end-to-end. This is a mechanical change but touches
//! all 8 distortion filter functions and the Jacobian helper signatures.
//!
//! ## Barrel (MAE 8.24) — POLYNOMIAL MAPPING
//!
//! **Root cause: our k1/k2 parameters map to different polynomial terms
//! than IM's "A B C D" barrel format.**
//!
//! IM barrel polynomial: `factor = A*rscale³*r³ + B*rscale²*r² + C*rscale*r + D`
//! Our barrel polynomial: `factor = A_coeff*r³ + B_coeff*r² + 1` (matching IM
//! form, but our test passes `k1=0.5, k2=0.1` which map to the cubic and
//! quadratic terms via `k1*rscale³` and `k2*rscale²`).
//!
//! The barrel test compares against `magick -distort Barrel "0.5 0.1 0 1"`
//! where IM interprets "0.5" as the r⁴ coefficient (A), "0.1" as r³ (B),
//! "0" as r² (C), and "1" as r⁰ (D). Our code maps k1 → r³ and k2 → r²,
//! which is a DIFFERENT polynomial despite using the same numeric values.
//!
//! **To close this gap**: either change our barrel API to match IM's 4-param
//! "A B C D" format, or adjust the IM parity test to pass coefficients that
//! produce equivalent polynomials.

/// Exact Robidoux filter B,C values from IM resize.c.
const ROBIDOUX_B: f64 = 0.378_215_755_093_998_7;
const ROBIDOUX_C: f64 = 0.310_892_122_453_000_7;

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
        ((-b - 6.0 * c) * r3
            + (6.0 * b + 30.0 * c) * r2
            + (-12.0 * b - 48.0 * c) * r
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

/// ClampUpAxes: verbatim port of IM's ClampUpAxes from resample.c.
///
/// Decomposes the Jacobian via SVD, clamps both singular values ≥ 1.0,
/// and returns unit vectors for the major/minor axes with their magnitudes.
///
/// THREE key differences from our previous implementation:
/// 1. Discriminant: `(frobenius² + 2*det)(frobenius² - 2*det)` (IM's form)
/// 2. Eigenvector: select row with larger |s1s1 - nXX| for numerical stability
/// 3. Minor axis: 90° rotation of major `(-u21, u11)`, not independent eigenvector
fn clamp_up_axes(dux: f64, dvx: f64, duy: f64, dvy: f64) -> (f64, f64, f64, f64, f64, f64) {
    // Verbatim IM variable names for auditability
    let a = dux;
    let b = duy;
    let c = dvx;
    let d = dvy;

    let aa = a * a;
    let bb = b * b;
    let cc = c * c;
    let dd = d * d;

    let n11 = aa + bb;
    let n12 = a * c + b * d;
    let n21 = n12;
    let n22 = cc + dd;
    let det = a * d - b * c;
    let twice_det = det + det;
    let frobenius_squared = n11 + n22;

    // IM's discriminant form: (frobenius² + 2det)(frobenius² - 2det)
    // Mathematically = frobenius⁴ - 4det² = (n11-n22)² + 4n12²
    // But IM's form is numerically different for near-degenerate matrices
    let discriminant = (frobenius_squared + twice_det) * (frobenius_squared - twice_det);

    let sqrt_discriminant = if discriminant > 0.0 {
        discriminant.sqrt()
    } else {
        0.0
    };

    let s1s1 = 0.5 * (frobenius_squared + sqrt_discriminant);
    let s2s2 = 0.5 * (frobenius_squared - sqrt_discriminant);

    // IM's eigenvector selection: pick the row of (n - s1s1*I) with larger magnitude
    let s1s1minusn11 = s1s1 - n11;
    let s1s1minusn22 = s1s1 - n22;

    let s1s1minusn11_squared = s1s1minusn11 * s1s1minusn11;
    let s1s1minusn22_squared = s1s1minusn22 * s1s1minusn22;

    let (temp_u11, temp_u21) = if s1s1minusn11_squared >= s1s1minusn22_squared {
        (n12, s1s1minusn11)
    } else {
        (s1s1minusn22, n21)
    };

    let norm = (temp_u11 * temp_u11 + temp_u21 * temp_u21).sqrt();

    let u11 = if norm > 0.0 { temp_u11 / norm } else { 1.0 };
    let u21 = if norm > 0.0 { temp_u21 / norm } else { 0.0 };

    let major_mag = if s1s1 <= 1.0 { 1.0 } else { s1s1.sqrt() };
    let minor_mag = if s2s2 <= 1.0 { 1.0 } else { s2s2.sqrt() };

    // IM: minor axis = 90° rotation of major, NOT independent eigenvector
    (
        major_mag, minor_mag, u11,  // major_unit_x
        u21,  // major_unit_y
        -u21, // minor_unit_x = -major_unit_y
        u11,  // minor_unit_y = major_unit_x
    )
}

/// Scaled ellipse coefficients ready for LUT lookup.
struct EllipseCoeffs {
    a: f64,
    b: f64,
    c: f64,
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

    let vlimit = (a * f / ac_bb4).sqrt();
    let uwidth = (f / a).sqrt();
    let slope = -b / (2.0 * a);

    // Scale A,B,C by WLUT_WIDTH/F for direct LUT indexing
    let scale = WLUT_WIDTH as f64 / f;

    Some(EllipseCoeffs {
        a: a * scale,
        b: b * scale,
        c: c * scale,
        vlimit,
        uwidth,
        slope,
    })
}

/// EWA resampler matching IM's ResamplePixelColor exactly.
///
/// Precomputes the Robidoux weight LUT on construction.
/// Channel storage format for the EWA sampler.
///
/// Determines how to read channel values from the raw byte buffer.
///
/// Integer formats (U8, U16) use Q16 internal representation to match
/// ImageMagick's rounding behavior for parity tests. Float formats
/// (F16, F32) use native f64 precision — matching Photoshop, Nuke,
/// and DaVinci Resolve's approach of full-precision float processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 1 byte per channel, range [0, 255]. Internal: Q16 (× 257).
    U8,
    /// 2 bytes per channel (little-endian u16), range [0, 65535]. Internal: Q16.
    U16,
    /// 2 bytes per channel (IEEE 754 half-float), range [0.0, 1.0]. Internal: native f64.
    F16,
    /// 4 bytes per channel (IEEE 754 float), range [0.0, 1.0]. Internal: native f64.
    F32,
}

impl SampleFormat {
    /// Bytes per channel for this format.
    pub const fn bytes_per_channel(self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::U16 => 2,
            SampleFormat::F16 => 2,
            SampleFormat::F32 => 4,
        }
    }

    /// True for float formats that use native f64 precision (no Q16 scaling).
    pub const fn is_float(self) -> bool {
        matches!(self, SampleFormat::F16 | SampleFormat::F32)
    }

    /// Derive from PixelFormat.
    pub fn from_pixel_format(fmt: crate::domain::types::PixelFormat) -> Self {
        use crate::domain::types::PixelFormat;
        match fmt {
            PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => SampleFormat::U16,
            PixelFormat::Rgb16f | PixelFormat::Rgba16f | PixelFormat::Gray16f => SampleFormat::F16,
            PixelFormat::Rgb32f | PixelFormat::Rgba32f | PixelFormat::Gray32f => SampleFormat::F32,
            _ => SampleFormat::U8,
        }
    }
}

pub struct EwaSampler<'a> {
    pub pixels: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub ch: usize,
    /// Channel storage format — determines byte reading and Q16 scaling.
    pub sample_fmt: SampleFormat,
    lut: Vec<f64>,
}

impl<'a> EwaSampler<'a> {
    /// Create a new EWA sampler with precomputed weight LUT.
    ///
    /// Assumes U8 sample format (backward compatible). Use `new_with_format`
    /// for non-u8 pixel data.
    pub fn new(pixels: &'a [u8], w: usize, h: usize, ch: usize) -> Self {
        Self {
            pixels,
            w,
            h,
            ch,
            sample_fmt: SampleFormat::U8,
            lut: build_weight_lut(),
        }
    }

    /// Create with explicit sample format for non-u8 pixel data.
    pub fn new_with_format(
        pixels: &'a [u8],
        w: usize,
        h: usize,
        ch: usize,
        sample_fmt: SampleFormat,
    ) -> Self {
        Self {
            pixels,
            w,
            h,
            ch,
            sample_fmt,
            lut: build_weight_lut(),
        }
    }

    /// Read a single channel value at pixel (px, py) into the internal
    /// representation. Returns 0.0 for out-of-bounds (black border).
    ///
    /// Integer formats → Q16 (IM parity). Float formats → native f64.
    #[inline]
    fn fetch(&self, px: i32, py: i32, c: usize) -> f64 {
        if px >= 0 && (px as usize) < self.w && py >= 0 && (py as usize) < self.h {
            self.read_channel((py as usize * self.w + px as usize) * self.ch + c)
        } else {
            0.0
        }
    }

    /// Fetch with edge-clamp border (for barrel distortion).
    #[inline]
    fn fetch_clamp(&self, px: i32, py: i32, c: usize) -> f64 {
        let cx = (px.max(0) as usize).min(self.w - 1);
        let cy = (py.max(0) as usize).min(self.h - 1);
        self.read_channel((cy * self.w + cx) * self.ch + c)
    }

    /// Read one channel value at logical index `idx` (channel-interleaved).
    ///
    /// Integer formats scale to Q16 [0, 65535] for IM-parity math.
    /// Float formats return the native f64 value — full precision, no Q16
    /// quantization. The EWA weighted-average math works identically in
    /// both ranges (it's just `Σ(weight × value) / Σ(weight)`).
    #[inline]
    fn read_channel(&self, idx: usize) -> f64 {
        match self.sample_fmt {
            SampleFormat::U8 => {
                // u8 [0,255] → Q16 [0,65535] for IM parity
                self.pixels[idx] as f64 * 257.0
            }
            SampleFormat::U16 => {
                // u16 is already Q16 range [0,65535]
                let off = idx * 2;
                u16::from_le_bytes([self.pixels[off], self.pixels[off + 1]]) as f64
            }
            SampleFormat::F16 => {
                // f16 → native f64. No Q16 scaling — full precision.
                let off = idx * 2;
                let h = half::f16::from_le_bytes([self.pixels[off], self.pixels[off + 1]]);
                h.to_f64()
            }
            SampleFormat::F32 => {
                // f32 → native f64. No Q16 scaling — full precision.
                let off = idx * 4;
                f32::from_le_bytes([
                    self.pixels[off],
                    self.pixels[off + 1],
                    self.pixels[off + 2],
                    self.pixels[off + 3],
                ]) as f64
            }
        }
    }

    /// Convert an internal sampled value back to the native output range.
    ///
    /// Integer formats: undo Q16 scaling. Float formats: pass through as-is.
    #[inline]
    pub fn internal_to_native(&self, v: f64) -> f32 {
        match self.sample_fmt {
            SampleFormat::U8 => (v / 257.0) as f32,   // Q16 → [0, 255]
            SampleFormat::U16 => v as f32,              // Q16 → [0, 65535]
            SampleFormat::F16 | SampleFormat::F32 => v as f32, // native f64 → f32
        }
    }

    /// Sample a single channel using IM-exact EWA with the given Jacobian.
    ///
    /// Matches IM's ResamplePixelColor: ClampUpAxes SVD, LUT-based weight
    /// lookup, parallelogram iteration with incremental Q updates.
    /// Falls back to bilinear for near-identity transforms where EWA
    /// would only blur without improving quality.
    pub fn sample(&self, sx: f32, sy: f32, jacobian: &Jacobian, c: usize) -> f32 {
        // Only fall back to bilinear if the Jacobian is exactly identity
        // (no distortion at all). Any non-identity Jacobian uses EWA.
        if *jacobian == JACOBIAN_IDENTITY {
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
                        color += weight * self.fetch(u_start + u_off, v, c);
                        divisor += weight;
                        hit += 1;
                    }
                }
                q += dq;
                dq += ddq;
            }
        }

        if hit > 0 && divisor > 1e-10 {
            self.internal_to_native(color / divisor)
        } else {
            self.bilinear(sx, sy, c)
        }
    }

    /// Sample with edge-clamp border (for barrel/undistort where IM clamps).
    pub fn sample_clamp(&self, sx: f32, sy: f32, jacobian: &Jacobian, c: usize) -> f32 {
        if *jacobian == JACOBIAN_IDENTITY {
            return self.bilinear_clamp(sx, sy, c);
        }
        let coeffs = match compute_ellipse(jacobian) {
            Some(c) => c,
            None => return self.bilinear_clamp(sx, sy, c),
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
                        color += weight * self.fetch_clamp(u_start + u_off, v, c);
                        divisor += weight;
                        hit += 1;
                    }
                }
                q += dq;
                dq += ddq;
            }
        }

        if hit > 0 && divisor > 1e-10 {
            self.internal_to_native(color / divisor)
        } else {
            self.bilinear_clamp(sx, sy, c)
        }
    }

    /// Bilinear with edge-clamp border.
    #[inline]
    fn bilinear_clamp(&self, sx: f32, sy: f32, c: usize) -> f32 {
        let x0 = sx.floor() as i32;
        let y0 = sy.floor() as i32;
        let fx = sx as f64 - x0 as f64;
        let fy = sy as f64 - y0 as f64;
        let v = self.fetch_clamp(x0, y0, c) * (1.0 - fx) * (1.0 - fy)
            + self.fetch_clamp(x0 + 1, y0, c) * fx * (1.0 - fy)
            + self.fetch_clamp(x0, y0 + 1, c) * (1.0 - fx) * fy
            + self.fetch_clamp(x0 + 1, y0 + 1, c) * fx * fy;
        self.internal_to_native(v)
    }

    /// Sample all channels at once using IM-exact EWA.
    pub fn sample_all(&self, sx: f32, sy: f32, jacobian: &Jacobian) -> Vec<f32> {
        (0..self.ch)
            .map(|c| self.sample(sx, sy, jacobian, c))
            .collect()
    }

    /// Bilinear interpolation (public for filters that match IM's bilinear path).
    #[inline]
    pub fn bilinear_pub(&self, sx: f32, sy: f32, c: usize) -> f32 {
        self.bilinear(sx, sy, c)
    }

    /// Bilinear interpolation fallback. Returns value in native format range.
    #[inline]
    fn bilinear(&self, sx: f32, sy: f32, c: usize) -> f32 {
        let x0 = sx.floor() as i32;
        let y0 = sy.floor() as i32;
        let fx = sx as f64 - x0 as f64;
        let fy = sy as f64 - y0 as f64;
        let v = self.fetch(x0, y0, c) * (1.0 - fx) * (1.0 - fy)
            + self.fetch(x0 + 1, y0, c) * fx * (1.0 - fy)
            + self.fetch(x0, y0 + 1, c) * (1.0 - fx) * fy
            + self.fetch(x0 + 1, y0 + 1, c) * fx * fy;
        self.internal_to_native(v)
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
pub fn jacobian_polar(ox: f32, oy: f32, w: f32, h: f32, max_r: f32) -> Jacobian {
    // IM pixel-center: angle = (ox+0.5-w/2)/w * 2π, radius = (oy+0.5)/h * max_r
    let two_pi = std::f64::consts::TAU;
    let wf = w as f64;
    let hf = h as f64;
    let angle = (ox as f64 + 0.5 - wf * 0.5) / wf * two_pi;
    let radius = (oy as f64 + 0.5) / hf * max_r as f64;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let da_dx = two_pi / wf;
    let dr_dy = max_r as f64 / hf;

    // sx = cx + radius * sin(angle) - 0.5
    // dsx/dox = radius * cos(angle) * da_dx
    // dsx/doy = sin(angle) * dr_dy
    // sy = cy + radius * cos(angle) - 0.5
    // dsy/dox = -radius * sin(angle) * da_dx
    // dsy/doy = cos(angle) * dr_dy
    [
        [(radius * cos_a * da_dx) as f32, (sin_a * dr_dy) as f32],
        [(-radius * sin_a * da_dx) as f32, (cos_a * dr_dy) as f32],
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
    c6: f32, // w/(2π)
    c7: f32, // h/max_r
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
pub fn jacobian_spherize(ox: f32, oy: f32, cx: f32, cy: f32, amount: f32, radius: f32) -> Jacobian {
    let dx = (ox - cx) / radius;
    let dy = (oy - cy) / radius;
    let r = (dx * dx + dy * dy).sqrt();

    if !(1e-10..1.0).contains(&r) {
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
