//! Axis-aligned rectangle type for region-based tile queries.

/// An axis-aligned rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }

    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// True if self fully contains other.
    pub fn contains(&self, other: &Rect) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// True if self and other overlap.
    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Compute the intersection of two rectangles. Returns None if no overlap.
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());

        if x < right && y < bottom {
            Some(Rect::new(x, y, right - x, bottom - y))
        } else {
            None
        }
    }

    /// Compute the bounding box containing both rectangles.
    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = self.right().max(other.right());
        let bottom = self.bottom().max(other.bottom());
        Rect::new(x, y, right - x, bottom - y)
    }

    /// Clamp this rect to image bounds (no expansion).
    pub fn clamp(&self, bounds_width: u32, bounds_height: u32) -> Rect {
        let x = self.x.min(bounds_width);
        let y = self.y.min(bounds_height);
        let right = self.right().min(bounds_width);
        let bottom = self.bottom().min(bounds_height);
        Rect::new(x, y, right.saturating_sub(x), bottom.saturating_sub(y))
    }

    /// Expand this rect uniformly on all sides, clamped to bounds.
    pub fn expand_uniform(&self, amount: u32, bounds_width: u32, bounds_height: u32) -> Rect {
        self.expand(&Overlap::uniform(amount), bounds_width, bounds_height)
    }

    /// Expand this rect asymmetrically, clamped to bounds.
    pub fn expand_asymmetric(
        &self,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        bounds_width: u32,
        bounds_height: u32,
    ) -> Rect {
        self.expand(
            &Overlap {
                left,
                top,
                right,
                bottom,
            },
            bounds_width,
            bounds_height,
        )
    }

    /// Expand this rect by the given overlap on each side, clamped to bounds.
    pub fn expand(&self, overlap: &Overlap, bounds_width: u32, bounds_height: u32) -> Rect {
        let x = self.x.saturating_sub(overlap.left);
        let y = self.y.saturating_sub(overlap.top);
        let right = (self.right() + overlap.right).min(bounds_width);
        let bottom = (self.bottom() + overlap.bottom).min(bounds_height);
        Rect::new(x, y, right - x, bottom - y)
    }

    /// Compute the set of rectangles covering `self` minus `other`.
    /// Returns up to 4 non-overlapping rectangles.
    ///
    /// ```text
    ///  ┌──────────────────┐
    ///  │       TOP        │
    ///  ├────┬────────┬────┤
    ///  │LEFT│ other  │RGHT│
    ///  ├────┴────────┴────┤
    ///  │      BOTTOM      │
    ///  └──────────────────┘
    /// ```
    pub fn difference(&self, other: &Rect) -> Vec<Rect> {
        let Some(inter) = self.intersection(other) else {
            return vec![*self];
        };

        let mut result = Vec::with_capacity(4);

        // Top strip
        if inter.y > self.y {
            result.push(Rect::new(self.x, self.y, self.width, inter.y - self.y));
        }
        // Bottom strip
        if inter.bottom() < self.bottom() {
            result.push(Rect::new(
                self.x,
                inter.bottom(),
                self.width,
                self.bottom() - inter.bottom(),
            ));
        }
        // Left strip (between top and bottom)
        if inter.x > self.x {
            result.push(Rect::new(self.x, inter.y, inter.x - self.x, inter.height));
        }
        // Right strip (between top and bottom)
        if inter.right() < self.right() {
            result.push(Rect::new(
                inter.right(),
                inter.y,
                self.right() - inter.right(),
                inter.height,
            ));
        }

        result
    }

    /// Compute the set of rectangles covering `self` minus all `others`.
    pub fn difference_all(&self, others: &[Rect]) -> Vec<Rect> {
        let mut remaining = vec![*self];
        for other in others {
            let mut next = Vec::new();
            for r in &remaining {
                next.extend(r.difference(other));
            }
            remaining = next;
        }
        remaining
    }

    /// Number of bytes needed to store pixels for this region at given bytes-per-pixel.
    pub fn pixel_bytes(&self, bpp: u32) -> usize {
        self.width as usize * self.height as usize * bpp as usize
    }
}

/// How much extra context a node needs from upstream per side.
#[derive(Debug, Clone, Copy, Default)]
pub struct Overlap {
    pub top: u32,
    pub bottom: u32,
    pub left: u32,
    pub right: u32,
}

impl Overlap {
    pub fn uniform(amount: u32) -> Self {
        Self {
            top: amount,
            bottom: amount,
            left: amount,
            right: amount,
        }
    }

    pub fn zero() -> Self {
        Self::default()
    }
}

/// Iterate over non-overlapping tiles covering a region.
///
/// Tiles are row-major (left→right, top→bottom). Edge tiles are clipped
/// to the image bounds — they may be smaller than `tile_w × tile_h`.
pub fn tiles(image_w: u32, image_h: u32, tile_w: u32, tile_h: u32) -> Vec<Rect> {
    let mut result = Vec::new();
    let mut y = 0u32;
    while y < image_h {
        let h = tile_h.min(image_h - y);
        let mut x = 0u32;
        while x < image_w {
            let w = tile_w.min(image_w - x);
            result.push(Rect::new(x, y, w, h));
            x += tile_w;
        }
        y += tile_h;
    }
    result
}

/// Extract a tile's f32 pixel data from a full image buffer.
///
/// `image` is `image_w × image_h × 4` floats. Returns `tile.width × tile.height × 4` floats.
pub fn extract_tile(image: &[f32], image_w: u32, tile: Rect) -> Vec<f32> {
    let stride = image_w as usize * 4;
    let tw = tile.width as usize * 4;
    let mut out = Vec::with_capacity(tile.width as usize * tile.height as usize * 4);
    for row in 0..tile.height as usize {
        let src = (tile.y as usize + row) * stride + tile.x as usize * 4;
        out.extend_from_slice(&image[src..src + tw]);
    }
    out
}

/// Place a tile's f32 pixel data into a full image buffer.
pub fn place_tile(image: &mut [f32], image_w: u32, tile: Rect, tile_data: &[f32]) {
    let stride = image_w as usize * 4;
    let tw = tile.width as usize * 4;
    for row in 0..tile.height as usize {
        let dst = (tile.y as usize + row) * stride + tile.x as usize * 4;
        let src = row * tw;
        image[dst..dst + tw].copy_from_slice(&tile_data[src..src + tw]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_covers_image() {
        let ts = tiles(256, 256, 64, 64);
        assert_eq!(ts.len(), 16); // 4x4 grid
        // All tiles cover the image
        let total_area: u64 = ts.iter().map(|t| t.area()).sum();
        assert_eq!(total_area, 256 * 256);
    }

    #[test]
    fn tiles_handles_non_divisible() {
        let ts = tiles(100, 100, 64, 64);
        assert_eq!(ts.len(), 4); // 2x2 grid
        assert_eq!(ts[0], Rect::new(0, 0, 64, 64));
        assert_eq!(ts[1], Rect::new(64, 0, 36, 64)); // clipped width
        assert_eq!(ts[2], Rect::new(0, 64, 64, 36)); // clipped height
        assert_eq!(ts[3], Rect::new(64, 64, 36, 36)); // both clipped
    }

    #[test]
    fn extract_place_roundtrip() {
        let w = 4u32;
        let h = 4u32;
        let image: Vec<f32> = (0..w * h * 4).map(|i| i as f32).collect();
        let tile = Rect::new(1, 1, 2, 2);
        let extracted = extract_tile(&image, w, tile);
        assert_eq!(extracted.len(), 2 * 2 * 4);

        let mut output = vec![0.0f32; (w * h * 4) as usize];
        place_tile(&mut output, w, tile, &extracted);
        // Check that the tile region matches
        for row in 0..2usize {
            for col in 0..2usize {
                for ch in 0..4usize {
                    let img_idx = ((1 + row) * w as usize + (1 + col)) * 4 + ch;
                    let tile_idx = (row * 2 + col) * 4 + ch;
                    assert_eq!(output[img_idx], extracted[tile_idx]);
                }
            }
        }
    }

    #[test]
    fn rect_basic() {
        let r = Rect::new(10, 20, 30, 40);
        assert_eq!(r.right(), 40);
        assert_eq!(r.bottom(), 60);
        assert_eq!(r.area(), 1200);
        assert!(!r.is_empty());
    }

    #[test]
    fn rect_empty() {
        assert!(Rect::new(0, 0, 0, 10).is_empty());
        assert!(Rect::new(0, 0, 10, 0).is_empty());
    }

    #[test]
    fn contains() {
        let outer = Rect::new(0, 0, 100, 100);
        let inner = Rect::new(10, 10, 20, 20);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
        assert!(outer.contains(&outer));
    }

    #[test]
    fn intersects() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 10, 10);
        let c = Rect::new(20, 20, 10, 10);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn intersection() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 10, 10);
        let i = a.intersection(&b).unwrap();
        assert_eq!(i, Rect::new(5, 5, 5, 5));
    }

    #[test]
    fn intersection_none() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(20, 20, 10, 10);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn union_rects() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 10, 10);
        let u = a.union(&b);
        assert_eq!(u, Rect::new(0, 0, 15, 15));
    }

    #[test]
    fn difference_no_overlap() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(20, 20, 10, 10);
        let d = a.difference(&b);
        assert_eq!(d, vec![a]);
    }

    #[test]
    fn difference_full_cover() {
        let a = Rect::new(5, 5, 10, 10);
        let b = Rect::new(0, 0, 100, 100);
        let d = a.difference(&b);
        assert!(d.is_empty());
    }

    #[test]
    fn difference_partial() {
        // a is 0,0 -> 20,20; b covers 5,5 -> 15,15
        let a = Rect::new(0, 0, 20, 20);
        let b = Rect::new(5, 5, 10, 10);
        let d = a.difference(&b);
        // Should produce 4 rects: top, bottom, left, right
        assert_eq!(d.len(), 4);
        // Total area should equal a.area() - intersection.area()
        let total: u64 = d.iter().map(|r| r.area()).sum();
        assert_eq!(total, 20 * 20 - 10 * 10);
    }

    #[test]
    fn difference_all_multiple() {
        let a = Rect::new(0, 0, 100, 100);
        let others = vec![Rect::new(0, 0, 50, 50), Rect::new(50, 50, 50, 50)];
        let d = a.difference_all(&others);
        let total: u64 = d.iter().map(|r| r.area()).sum();
        assert_eq!(total, 100 * 100 - 50 * 50 - 50 * 50);
    }

    #[test]
    fn expand_with_bounds() {
        let r = Rect::new(5, 5, 10, 10);
        let o = Overlap::uniform(3);
        let e = r.expand(&o, 100, 100);
        assert_eq!(e, Rect::new(2, 2, 16, 16));
    }

    #[test]
    fn expand_clamped_to_zero() {
        let r = Rect::new(1, 1, 10, 10);
        let o = Overlap::uniform(5);
        let e = r.expand(&o, 100, 100);
        assert_eq!(e.x, 0);
        assert_eq!(e.y, 0);
    }

    #[test]
    fn expand_clamped_to_bounds() {
        let r = Rect::new(90, 90, 10, 10);
        let o = Overlap::uniform(5);
        let e = r.expand(&o, 100, 100);
        assert_eq!(e.right(), 100);
        assert_eq!(e.bottom(), 100);
    }
}
