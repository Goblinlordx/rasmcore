use super::{ImageError, ImageInfo, Rgb, find_nearest, find_nearest_index};

/// Generate an N-color palette from an RGB8 image using the median cut algorithm.
///
/// Recursively splits the color-space bounding box along the axis of greatest range,
/// partitioning at the median. Each final box's average color becomes a palette entry.
///
/// `max_colors` must be 2..=256.
pub fn median_cut(
    pixels: &[u8],
    info: &ImageInfo,
    max_colors: usize,
) -> Result<Vec<Rgb>, ImageError> {
    if !(2..=256).contains(&max_colors) {
        return Err(ImageError::InvalidParameters(
            "max_colors must be 2..256".into(),
        ));
    }
    let n = (info.width * info.height) as usize;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }

    // Collect all pixels as (r, g, b) tuples
    let mut colors: Vec<[u8; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        colors.push([pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]]);
    }

    // Start with one box containing all colors
    let mut boxes: Vec<ColorBox> = vec![ColorBox::new(&mut colors, 0, n)];

    // Split until we have max_colors boxes (or can't split further)
    while boxes.len() < max_colors {
        // Find the box with the largest range to split
        let (split_idx, _) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.count > 1)
            .max_by_key(|(_, b)| b.longest_axis_range())
            .unwrap_or((0, &boxes[0]));

        if boxes[split_idx].count <= 1 {
            break; // Can't split any further
        }

        let b = boxes.remove(split_idx);
        let (left, right) = b.split(&mut colors);
        boxes.push(left);
        boxes.push(right);
    }

    // Average color per box = palette entry
    let palette: Vec<Rgb> = boxes.iter().map(|b| b.average(&colors)).collect();
    Ok(palette)
}

/// Map each pixel in an RGB8 image to the nearest color in the palette.
///
/// Returns a new RGB8 image with only palette colors.
pub fn quantize(pixels: &[u8], info: &ImageInfo, palette: &[Rgb]) -> Result<Vec<u8>, ImageError> {
    let n = (info.width * info.height) as usize;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if palette.is_empty() {
        return Err(ImageError::InvalidParameters("palette is empty".into()));
    }

    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;

        let nearest = find_nearest(r, g, b, palette);
        out[i * 3] = nearest.r;
        out[i * 3 + 1] = nearest.g;
        out[i * 3 + 2] = nearest.b;
    }
    Ok(out)
}

/// Map each pixel to a palette INDEX (0..palette.len()-1).
/// Useful for indexed/palette image formats (GIF, PNG8).
pub fn quantize_indexed(
    pixels: &[u8],
    info: &ImageInfo,
    palette: &[Rgb],
) -> Result<Vec<u8>, ImageError> {
    let n = (info.width * info.height) as usize;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if palette.is_empty() {
        return Err(ImageError::InvalidParameters("palette is empty".into()));
    }

    let mut indices = vec![0u8; n];
    for i in 0..n {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;

        indices[i] = find_nearest_index(r, g, b, palette) as u8;
    }
    Ok(indices)
}

// ─── Internals ─────────────────────────────────────────────────────────────

/// A bounding box in RGB color space for median cut.
struct ColorBox {
    start: usize,
    count: usize,
    r_min: u8,
    r_max: u8,
    g_min: u8,
    g_max: u8,
    b_min: u8,
    b_max: u8,
}

impl ColorBox {
    fn new(colors: &mut [[u8; 3]], start: usize, count: usize) -> Self {
        let slice = &colors[start..start + count];
        let (mut rmin, mut rmax) = (255u8, 0u8);
        let (mut gmin, mut gmax) = (255u8, 0u8);
        let (mut bmin, mut bmax) = (255u8, 0u8);
        for c in slice {
            rmin = rmin.min(c[0]);
            rmax = rmax.max(c[0]);
            gmin = gmin.min(c[1]);
            gmax = gmax.max(c[1]);
            bmin = bmin.min(c[2]);
            bmax = bmax.max(c[2]);
        }
        ColorBox {
            start,
            count,
            r_min: rmin,
            r_max: rmax,
            g_min: gmin,
            g_max: gmax,
            b_min: bmin,
            b_max: bmax,
        }
    }

    fn longest_axis_range(&self) -> u16 {
        let r_range = (self.r_max - self.r_min) as u16;
        let g_range = (self.g_max - self.g_min) as u16;
        let b_range = (self.b_max - self.b_min) as u16;
        r_range.max(g_range).max(b_range)
    }

    fn longest_axis(&self) -> usize {
        let r_range = self.r_max - self.r_min;
        let g_range = self.g_max - self.g_min;
        let b_range = self.b_max - self.b_min;
        if r_range >= g_range && r_range >= b_range {
            0
        } else if g_range >= b_range {
            1
        } else {
            2
        }
    }

    fn split(self, colors: &mut [[u8; 3]]) -> (ColorBox, ColorBox) {
        let axis = self.longest_axis();
        let slice = &mut colors[self.start..self.start + self.count];

        // Sort by the longest axis
        slice.sort_unstable_by_key(|c| c[axis]);

        let mid = self.count / 2;
        let left = ColorBox::new(colors, self.start, mid);
        let right = ColorBox::new(colors, self.start + mid, self.count - mid);
        (left, right)
    }

    fn average(&self, colors: &[[u8; 3]]) -> Rgb {
        let slice = &colors[self.start..self.start + self.count];
        let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
        for c in slice {
            sr += c[0] as u64;
            sg += c[1] as u64;
            sb += c[2] as u64;
        }
        let n = self.count as u64;
        Rgb {
            r: (sr / n) as u8,
            g: (sg / n) as u8,
            b: (sb / n) as u8,
        }
    }
}
