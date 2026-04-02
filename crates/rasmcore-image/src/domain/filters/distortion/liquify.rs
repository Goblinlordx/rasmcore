//! Liquify warp filters — push, pinch, expand, twirl, smooth.
//!
//! Each filter applies a single brush stroke that displaces pixels within a
//! circular radius. Push moves pixels in a direction, pinch contracts toward
//! center, expand pushes away, twirl rotates, and smooth relaxes displacement.
//!
//! All operations use Gaussian falloff for smooth brush edges.

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Gaussian weight: exp(-2 * (d/r)^2), drops to ~0.13 at edge
#[inline]
fn gaussian_weight(dist: f32, radius: f32) -> f32 {
    let t = dist / radius;
    (-2.0 * t * t).exp()
}

// ─── Liquify Push ─────────────────────────────────────────────────────────────

/// Push pixels in a direction within a circular brush.
///
/// Displaces pixels along (direction_x, direction_y) with Gaussian falloff.
/// Strength controls magnitude: 1.0 = full push, 0.5 = half.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "liquify_push",
    category = "distortion",
    group = "liquify",
    reference = "Liquify push/forward warp"
)]
pub struct LiquifyPushParams {
    /// Brush center X (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    /// Brush center Y (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
    /// Brush radius in pixels
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0)]
    pub radius: f32,
    /// Push strength (0.0–1.0)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
    /// Push direction X in pixels
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 20.0)]
    pub direction_x: f32,
    /// Push direction Y in pixels
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0)]
    pub direction_y: f32,
}

impl CpuFilter for LiquifyPushParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;
        if is_16bit(info.format) {
            let full = Rect::new(0, 0, info.width, info.height);
            let pixels = upstream(full)?;
            let cfg = self.clone();
            return process_via_8bit(&pixels, info, |px, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                cfg.compute(r, &mut |_| Ok(px.to_vec()), i8)
            });
        }

        let cx = self.center_x * info.width as f32;
        let cy = self.center_y * info.height as f32;
        let radius = self.radius;
        let strength = self.strength;
        let dir_x = self.direction_x;
        let dir_y = self.direction_y;
        let overlap = (dir_x.abs().max(dir_y.abs()) * strength + 1.0).ceil() as u32;

        apply_distortion(
            request, upstream, info,
            DistortionOverlap::Uniform(overlap),
            DistortionSampling::Bilinear,
            &|xf, yf| {
                let dx = xf - cx;
                let dy = yf - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= radius {
                    (xf, yf)
                } else {
                    let w = gaussian_weight(dist, radius) * strength;
                    // Inverse: source = output - displacement
                    (xf - dir_x * w, yf - dir_y * w)
                }
            },
            &|_, _| [[1.0, 0.0], [0.0, 1.0]],
        )
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = (self.direction_x.abs().max(self.direction_y.abs()) * self.strength + 1.0).ceil() as u32;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

// ─── Liquify Pinch ────────────────────────────────────────────────────────────

/// Contract pixels toward brush center (pinch/pucker).
///
/// Pixels within radius are displaced toward the center point with Gaussian falloff.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "liquify_pinch",
    category = "distortion",
    group = "liquify",
    reference = "Liquify pinch/pucker warp"
)]
pub struct LiquifyPinchParams {
    /// Brush center X (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    /// Brush center Y (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
    /// Brush radius in pixels
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0)]
    pub radius: f32,
    /// Pinch strength (0.0–1.0)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
}

impl CpuFilter for LiquifyPinchParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;
        if is_16bit(info.format) {
            let full = Rect::new(0, 0, info.width, info.height);
            let pixels = upstream(full)?;
            let cfg = self.clone();
            return process_via_8bit(&pixels, info, |px, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                cfg.compute(r, &mut |_| Ok(px.to_vec()), i8)
            });
        }

        let cx = self.center_x * info.width as f32;
        let cy = self.center_y * info.height as f32;
        let radius = self.radius;
        let strength = self.strength;
        let overlap = (radius * strength).ceil() as u32;

        apply_distortion(
            request, upstream, info,
            DistortionOverlap::Uniform(overlap),
            DistortionSampling::Bilinear,
            &|xf, yf| {
                let dx = xf - cx;
                let dy = yf - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= radius || dist < 1e-6 {
                    (xf, yf)
                } else {
                    let w = gaussian_weight(dist, radius) * strength;
                    // Pinch forward: new_pos = pos * (1 - w) + center * w
                    // Inverse: pos = (new_pos - center * w) / (1 - w)
                    let s = 1.0 / (1.0 - w).max(0.01);
                    (
                        cx + (xf - cx) * s,
                        cy + (yf - cy) * s,
                    )
                }
            },
            &|_, _| [[1.0, 0.0], [0.0, 1.0]],
        )
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = (self.radius * self.strength).ceil() as u32;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

// ─── Liquify Expand ───────────────────────────────────────────────────────────

/// Expand pixels away from brush center (bloat/punch).
///
/// Opposite of pinch — pixels within radius are pushed outward.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "liquify_expand",
    category = "distortion",
    group = "liquify",
    reference = "Liquify expand/bloat warp"
)]
pub struct LiquifyExpandParams {
    /// Brush center X (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    /// Brush center Y (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
    /// Brush radius in pixels
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0)]
    pub radius: f32,
    /// Expand strength (0.0–1.0)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
}

impl CpuFilter for LiquifyExpandParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;
        if is_16bit(info.format) {
            let full = Rect::new(0, 0, info.width, info.height);
            let pixels = upstream(full)?;
            let cfg = self.clone();
            return process_via_8bit(&pixels, info, |px, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                cfg.compute(r, &mut |_| Ok(px.to_vec()), i8)
            });
        }

        let cx = self.center_x * info.width as f32;
        let cy = self.center_y * info.height as f32;
        let radius = self.radius;
        let strength = self.strength;
        let overlap = (radius * strength).ceil() as u32;

        apply_distortion(
            request, upstream, info,
            DistortionOverlap::Uniform(overlap),
            DistortionSampling::Bilinear,
            &|xf, yf| {
                let dx = xf - cx;
                let dy = yf - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= radius || dist < 1e-6 {
                    (xf, yf)
                } else {
                    let w = gaussian_weight(dist, radius) * strength;
                    // Expand: displace away from center
                    // Forward: new_pos = pos + (pos - center) * w = pos * (1 + w) - center * w
                    // Inverse: pos = (new_pos + center * w) / (1 + w)
                    let inv = 1.0 / (1.0 + w);
                    (
                        (xf + cx * w) * inv,
                        (yf + cy * w) * inv,
                    )
                }
            },
            &|_, _| [[1.0, 0.0], [0.0, 1.0]],
        )
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = (self.radius * self.strength).ceil() as u32;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

// ─── Liquify Twirl ────────────────────────────────────────────────────────────

/// Rotate pixels around brush center.
///
/// Pixels within radius are rotated by `angle` (in degrees) with Gaussian falloff.
/// Positive = clockwise, negative = counter-clockwise.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "liquify_twirl",
    category = "distortion",
    group = "liquify",
    reference = "Liquify twirl/rotate warp"
)]
pub struct LiquifyTwirlParams {
    /// Brush center X (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    /// Brush center Y (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
    /// Brush radius in pixels
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0)]
    pub radius: f32,
    /// Twirl strength (0.0–1.0)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
    /// Rotation angle in degrees (positive = clockwise)
    #[param(min = -360.0, max = 360.0, step = 5.0, default = 45.0)]
    pub angle: f32,
}

impl CpuFilter for LiquifyTwirlParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;
        if is_16bit(info.format) {
            let full = Rect::new(0, 0, info.width, info.height);
            let pixels = upstream(full)?;
            let cfg = self.clone();
            return process_via_8bit(&pixels, info, |px, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                cfg.compute(r, &mut |_| Ok(px.to_vec()), i8)
            });
        }

        let cx = self.center_x * info.width as f32;
        let cy = self.center_y * info.height as f32;
        let radius = self.radius;
        let strength = self.strength;
        let angle_rad = self.angle.to_radians();
        let overlap = (radius * strength * (angle_rad.abs().min(1.0))).ceil() as u32 + 1;

        apply_distortion(
            request, upstream, info,
            DistortionOverlap::Uniform(overlap),
            DistortionSampling::Bilinear,
            &|xf, yf| {
                let dx = xf - cx;
                let dy = yf - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= radius || dist < 1e-6 {
                    (xf, yf)
                } else {
                    let w = gaussian_weight(dist, radius) * strength;
                    // Twirl: rotate by -angle*w to get the inverse (source coordinates)
                    let theta = -angle_rad * w;
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    (
                        cx + dx * cos_t - dy * sin_t,
                        cy + dx * sin_t + dy * cos_t,
                    )
                }
            },
            &|_, _| [[1.0, 0.0], [0.0, 1.0]],
        )
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = (self.radius * self.strength * (self.angle.to_radians().abs().min(1.0))).ceil() as u32 + 1;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

// ─── Liquify Smooth ───────────────────────────────────────────────────────────

/// Relax/attenuate displacement within brush radius.
///
/// Pushes pixels back toward their original position, effectively undoing
/// previous liquify strokes. Strength 1.0 = full restoration to identity,
/// 0.5 = halfway between current and original position.
///
/// As a standalone filter, this blends the upstream image toward an
/// unwarped center region (minor smoothing effect). Most useful in the
/// displacement field accumulation API where it directly attenuates the field.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "liquify_smooth",
    category = "distortion",
    group = "liquify",
    reference = "Liquify smooth/reconstruct"
)]
pub struct LiquifySmoothParams {
    /// Brush center X (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_x: f32,
    /// Brush center Y (0.0–1.0 normalized)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub center_y: f32,
    /// Brush radius in pixels
    #[param(min = 1.0, max = 500.0, step = 1.0, default = 50.0)]
    pub radius: f32,
    /// Smooth strength (0.0–1.0, 1.0 = full restore)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub strength: f32,
}

impl CpuFilter for LiquifySmoothParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        validate_format(info.format)?;
        if is_16bit(info.format) {
            let full = Rect::new(0, 0, info.width, info.height);
            let pixels = upstream(full)?;
            let cfg = self.clone();
            return process_via_8bit(&pixels, info, |px, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                cfg.compute(r, &mut |_| Ok(px.to_vec()), i8)
            });
        }

        // As a pixel filter, smooth just returns identity mapping (source = output).
        // The smooth effect only manifests in the displacement field accumulation API.
        // For the filter path: blend upstream toward identity (pass-through).
        apply_distortion(
            request, upstream, info,
            DistortionOverlap::Uniform(0),
            DistortionSampling::Bilinear,
            &|xf, yf| (xf, yf), // Identity — no displacement
            &|_, _| [[1.0, 0.0], [0.0, 1.0]],
        )
    }
}

// ─── Displacement Field API ───────────────────────────────────────────────────
//
// These standalone functions modify displacement fields directly. Used by
// interactive liquify tools that accumulate multiple brush strokes into a
// single displacement field, then render via displacement_map.

/// Apply a liquify push stroke to a displacement field.
///
/// Modifies `map_x` and `map_y` in-place. Fields must be `width * height` f32 slices
/// containing absolute source coordinates (identity = pixel's own position).
pub fn liquify_field_push(
    map_x: &mut [f32],
    map_y: &mut [f32],
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    radius: f32,
    strength: f32,
    direction_x: f32,
    direction_y: f32,
) {
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < radius {
                let wt = gaussian_weight(dist, radius) * strength;
                let idx = y * w + x;
                map_x[idx] -= direction_x * wt;
                map_y[idx] -= direction_y * wt;
            }
        }
    }
}

/// Apply a liquify pinch stroke to a displacement field.
pub fn liquify_field_pinch(
    map_x: &mut [f32],
    map_y: &mut [f32],
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    radius: f32,
    strength: f32,
) {
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < radius && dist > 1e-6 {
                let wt = gaussian_weight(dist, radius) * strength;
                let idx = y * w + x;
                // Pull source coordinates toward center
                map_x[idx] += (center_x - map_x[idx]) * wt;
                map_y[idx] += (center_y - map_y[idx]) * wt;
            }
        }
    }
}

/// Apply a liquify expand stroke to a displacement field.
pub fn liquify_field_expand(
    map_x: &mut [f32],
    map_y: &mut [f32],
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    radius: f32,
    strength: f32,
) {
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < radius && dist > 1e-6 {
                let wt = gaussian_weight(dist, radius) * strength;
                let idx = y * w + x;
                // Push source coordinates away from center
                map_x[idx] += (map_x[idx] - center_x) * wt;
                map_y[idx] += (map_y[idx] - center_y) * wt;
            }
        }
    }
}

/// Apply a liquify twirl stroke to a displacement field.
pub fn liquify_field_twirl(
    map_x: &mut [f32],
    map_y: &mut [f32],
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    radius: f32,
    strength: f32,
    angle_degrees: f32,
) {
    let w = width as usize;
    let h = height as usize;
    let angle_rad = angle_degrees.to_radians();
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < radius && dist > 1e-6 {
                let wt = gaussian_weight(dist, radius) * strength;
                let idx = y * w + x;
                // Rotate source coordinates around center
                let sdx = map_x[idx] - center_x;
                let sdy = map_y[idx] - center_y;
                let theta = -angle_rad * wt;
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                map_x[idx] = center_x + sdx * cos_t - sdy * sin_t;
                map_y[idx] = center_y + sdx * sin_t + sdy * cos_t;
            }
        }
    }
}

/// Apply smooth/relax to a displacement field — attenuate toward identity.
///
/// Blends each pixel's source coordinate toward its own position (identity mapping).
/// Strength 1.0 fully restores to identity within the brush; 0.5 = halfway.
pub fn liquify_field_smooth(
    map_x: &mut [f32],
    map_y: &mut [f32],
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    radius: f32,
    strength: f32,
) {
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < radius {
                let wt = gaussian_weight(dist, radius) * strength;
                let idx = y * w + x;
                // Blend toward identity: source → pixel's own position
                map_x[idx] += (x as f32 - map_x[idx]) * wt;
                map_y[idx] += (y as f32 - map_y[idx]) * wt;
            }
        }
    }
}

/// Create an identity displacement field (each pixel maps to itself).
pub fn identity_field(width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let mut map_x = Vec::with_capacity(n);
    let mut map_y = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            map_x.push(x as f32);
            map_y.push(y as f32);
        }
    }
    (map_x, map_y)
}
