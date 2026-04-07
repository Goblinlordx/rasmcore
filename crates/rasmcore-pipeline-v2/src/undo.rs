//! Tile-based stroke undo stack.
//!
//! Captures only tiles overlapping a stroke's bounding box before rendering.
//! Undo restores pre-stroke tiles. Memory-bounded with oldest-first eviction.

use std::collections::HashMap;
use crate::rect::Rect;

/// Tile size in pixels (64x64).
pub const TILE_SIZE: u32 = 64;

/// A sparse set of tiles captured from a pixel buffer.
///
/// Only tiles that overlap a given bounding rect are captured.
/// Each tile is 64x64 f32 RGBA (16,384 floats = 65,536 bytes).
#[derive(Clone)]
pub struct TileSet {
    /// Map from tile coordinates (tx, ty) to tile pixel data.
    pub tiles: HashMap<(u32, u32), Vec<f32>>,
    /// Source image dimensions (for bounds checking during restore).
    pub source_width: u32,
    pub source_height: u32,
}

impl TileSet {
    /// Capture tiles from a pixel buffer that overlap the given bounding rect.
    pub fn capture(pixels: &[f32], width: u32, height: u32, bounds: Rect) -> Self {
        let mut tiles = HashMap::new();

        // Compute tile range overlapping bounds
        let tx_min = bounds.x / TILE_SIZE;
        let ty_min = bounds.y / TILE_SIZE;
        let tx_max = (bounds.x + bounds.width).saturating_sub(1) / TILE_SIZE;
        let ty_max = (bounds.y + bounds.height).saturating_sub(1) / TILE_SIZE;

        for ty in ty_min..=ty_max {
            for tx in tx_min..=tx_max {
                let tile_x = tx * TILE_SIZE;
                let tile_y = ty * TILE_SIZE;
                if tile_x >= width || tile_y >= height {
                    continue;
                }

                // Extract tile pixels
                let tw = TILE_SIZE.min(width - tile_x);
                let th = TILE_SIZE.min(height - tile_y);
                let mut tile_data = vec![0.0f32; (TILE_SIZE * TILE_SIZE * 4) as usize];

                for row in 0..th {
                    let src_y = tile_y + row;
                    let src_offset = ((src_y * width + tile_x) * 4) as usize;
                    let dst_offset = ((row * TILE_SIZE) * 4) as usize;
                    let copy_len = (tw * 4) as usize;
                    if src_offset + copy_len <= pixels.len() {
                        tile_data[dst_offset..dst_offset + copy_len]
                            .copy_from_slice(&pixels[src_offset..src_offset + copy_len]);
                    }
                }

                tiles.insert((tx, ty), tile_data);
            }
        }

        TileSet { tiles, source_width: width, source_height: height }
    }

    /// Restore captured tiles back into a pixel buffer.
    pub fn restore(&self, pixels: &mut [f32], width: u32, height: u32) {
        for (&(tx, ty), tile_data) in &self.tiles {
            let tile_x = tx * TILE_SIZE;
            let tile_y = ty * TILE_SIZE;
            if tile_x >= width || tile_y >= height {
                continue;
            }

            let tw = TILE_SIZE.min(width - tile_x);
            let th = TILE_SIZE.min(height - tile_y);

            for row in 0..th {
                let dst_y = tile_y + row;
                let dst_offset = ((dst_y * width + tile_x) * 4) as usize;
                let src_offset = ((row * TILE_SIZE) * 4) as usize;
                let copy_len = (tw * 4) as usize;
                if dst_offset + copy_len <= pixels.len() {
                    pixels[dst_offset..dst_offset + copy_len]
                        .copy_from_slice(&tile_data[src_offset..src_offset + copy_len]);
                }
            }
        }
    }

    /// Memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.tiles.len() * (TILE_SIZE * TILE_SIZE * 4) as usize * std::mem::size_of::<f32>()
    }
}

/// A single undo entry representing one stroke's pre-state.
pub struct UndoEntry {
    /// Pre-stroke tile snapshot (only affected tiles).
    pub pre_tiles: TileSet,
    /// Post-stroke tile snapshot (for redo).
    pub post_tiles: Option<TileSet>,
    /// Bounding rect of the stroke.
    pub bounds: Rect,
}

/// Tile-based stroke undo stack with memory budget.
pub struct UndoStack {
    entries: Vec<UndoEntry>,
    /// Current position in the stack (entries[..position] are undoable).
    position: usize,
    /// Maximum total memory for tile snapshots (bytes).
    memory_budget: usize,
    /// Current total memory usage.
    memory_used: usize,
}

impl UndoStack {
    /// Create an undo stack with the given memory budget in bytes.
    pub fn new(memory_budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            position: 0,
            memory_budget,
            memory_used: 0,
        }
    }

    /// Push a stroke's pre-state onto the stack.
    ///
    /// Call this BEFORE rendering the stroke. Captures tiles from the current
    /// pixel buffer that overlap the stroke bounding rect.
    pub fn push_stroke(&mut self, pixels: &[f32], width: u32, height: u32, bounds: Rect) {
        // Discard any redo entries beyond current position
        while self.entries.len() > self.position {
            let removed = self.entries.pop().unwrap();
            self.memory_used -= removed.pre_tiles.size_bytes();
            if let Some(post) = &removed.post_tiles {
                self.memory_used -= post.size_bytes();
            }
        }

        let pre_tiles = TileSet::capture(pixels, width, height, bounds);
        let entry_size = pre_tiles.size_bytes();

        // Evict oldest entries until within budget
        while self.memory_used + entry_size > self.memory_budget && !self.entries.is_empty() {
            let removed = self.entries.remove(0);
            self.memory_used -= removed.pre_tiles.size_bytes();
            if let Some(post) = &removed.post_tiles {
                self.memory_used -= post.size_bytes();
            }
            self.position = self.position.saturating_sub(1);
        }

        self.memory_used += entry_size;
        self.entries.push(UndoEntry {
            pre_tiles,
            post_tiles: None,
            bounds,
        });
        self.position = self.entries.len();
    }

    /// Record the post-stroke state (call AFTER rendering the stroke).
    /// Needed for redo — captures what the stroke produced.
    pub fn record_post_state(&mut self, pixels: &[f32], width: u32, height: u32) {
        if self.position > 0 {
            let entry = &mut self.entries[self.position - 1];
            let post_tiles = TileSet::capture(pixels, width, height, entry.bounds);
            self.memory_used += post_tiles.size_bytes();
            entry.post_tiles = Some(post_tiles);
        }
    }

    /// Undo the last stroke. Returns the bounding rect of affected tiles.
    ///
    /// Restores pre-stroke tiles into the pixel buffer.
    pub fn undo(&mut self, pixels: &mut [f32], width: u32, height: u32) -> Option<Rect> {
        if self.position == 0 {
            return None;
        }
        self.position -= 1;
        let entry = &self.entries[self.position];
        entry.pre_tiles.restore(pixels, width, height);
        Some(entry.bounds)
    }

    /// Redo the last undone stroke. Returns the bounding rect of affected tiles.
    ///
    /// Restores post-stroke tiles into the pixel buffer.
    pub fn redo(&mut self, pixels: &mut [f32], width: u32, height: u32) -> Option<Rect> {
        if self.position >= self.entries.len() {
            return None;
        }
        let entry = &self.entries[self.position];
        if let Some(post_tiles) = &entry.post_tiles {
            post_tiles.restore(pixels, width, height);
            self.position += 1;
            Some(entry.bounds)
        } else {
            None
        }
    }

    /// Clear all undo/redo history.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.position = 0;
        self.memory_used = 0;
    }

    /// Set memory budget in bytes.
    pub fn set_memory_budget(&mut self, budget: usize) {
        self.memory_budget = budget;
        // Evict if over new budget
        while self.memory_used > self.memory_budget && !self.entries.is_empty() {
            let removed = self.entries.remove(0);
            self.memory_used -= removed.pre_tiles.size_bytes();
            if let Some(post) = &removed.post_tiles {
                self.memory_used -= post.size_bytes();
            }
            self.position = self.position.saturating_sub(1);
        }
    }

    /// Number of undoable entries.
    pub fn undo_count(&self) -> usize {
        self.position
    }

    /// Number of redoable entries.
    pub fn redo_count(&self) -> usize {
        self.entries.len() - self.position
    }

    /// Current memory usage in bytes.
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: u32, h: u32, color: [f32; 4]) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut px = Vec::with_capacity(n * 4);
        for _ in 0..n {
            px.extend_from_slice(&color);
        }
        px
    }

    #[test]
    fn tile_capture_and_restore_roundtrip() {
        let w = 128;
        let h = 128;
        let original = solid_image(w, h, [0.5, 0.3, 0.1, 1.0]);
        let bounds = Rect::new(10, 10, 50, 50);
        let tiles = TileSet::capture(&original, w, h, bounds);

        // Modify the image
        let mut modified = solid_image(w, h, [1.0, 1.0, 1.0, 1.0]);

        // Restore — affected area should match original
        tiles.restore(&mut modified, w, h);

        // Check a pixel inside the captured area
        let idx = ((20 * w + 20) * 4) as usize;
        assert!((modified[idx] - 0.5).abs() < 1e-6, "restored pixel should be 0.5");
    }

    #[test]
    fn undo_restores_pre_stroke_state() {
        let w = 128;
        let h = 128;
        let mut canvas = solid_image(w, h, [0.0, 0.0, 0.0, 1.0]);
        let bounds = Rect::new(0, 0, 64, 64);

        let mut stack = UndoStack::new(10 * 1024 * 1024);

        // Push pre-stroke state
        stack.push_stroke(&canvas, w, h, bounds);

        // Simulate stroke: paint white in bounds
        for y in 0..64u32 {
            for x in 0..64u32 {
                let i = ((y * w + x) * 4) as usize;
                canvas[i] = 1.0;
                canvas[i + 1] = 1.0;
                canvas[i + 2] = 1.0;
            }
        }

        stack.record_post_state(&canvas, w, h);

        // Undo
        let result = stack.undo(&mut canvas, w, h);
        assert!(result.is_some());

        // Verify pixel is back to black
        assert!((canvas[0] - 0.0).abs() < 1e-6, "should be black after undo");
    }

    #[test]
    fn redo_restores_post_stroke_state() {
        let w = 128;
        let h = 128;
        let mut canvas = solid_image(w, h, [0.0, 0.0, 0.0, 1.0]);
        let bounds = Rect::new(0, 0, 64, 64);

        let mut stack = UndoStack::new(10 * 1024 * 1024);
        stack.push_stroke(&canvas, w, h, bounds);

        // Paint white
        for y in 0..64u32 {
            for x in 0..64u32 {
                let i = ((y * w + x) * 4) as usize;
                canvas[i] = 1.0; canvas[i + 1] = 1.0; canvas[i + 2] = 1.0;
            }
        }
        stack.record_post_state(&canvas, w, h);

        // Undo then redo
        stack.undo(&mut canvas, w, h);
        assert!((canvas[0] - 0.0).abs() < 1e-6);
        stack.redo(&mut canvas, w, h);
        assert!((canvas[0] - 1.0).abs() < 1e-6, "should be white after redo");
    }

    #[test]
    fn memory_budget_evicts_oldest() {
        let w = 128;
        let h = 128;
        let canvas = solid_image(w, h, [0.5, 0.5, 0.5, 1.0]);
        let bounds = Rect::new(0, 0, 128, 128);

        // Small budget: ~1 entry worth
        let tile_size = TileSet::capture(&canvas, w, h, bounds).size_bytes();
        let mut stack = UndoStack::new(tile_size + 100);

        stack.push_stroke(&canvas, w, h, bounds);
        assert_eq!(stack.undo_count(), 1);

        // Second push should evict the first
        stack.push_stroke(&canvas, w, h, bounds);
        assert_eq!(stack.undo_count(), 1, "oldest should be evicted");
    }

    #[test]
    fn roundtrip_three_strokes_undo_all() {
        let w = 128;
        let h = 128;
        let original = solid_image(w, h, [0.0, 0.0, 0.0, 1.0]);
        let mut canvas = original.clone();

        let mut stack = UndoStack::new(50 * 1024 * 1024);

        // Draw 3 strokes in different areas
        for i in 0..3 {
            let x = i * 40;
            let bounds = Rect::new(x, 0, 30, 30);
            stack.push_stroke(&canvas, w, h, bounds);
            // Paint stroke area
            for sy in 0..30u32 {
                for sx in 0..30u32 {
                    let px = x + sx;
                    let py = sy;
                    if px < w && py < h {
                        let idx = ((py * w + px) * 4) as usize;
                        canvas[idx] = (i + 1) as f32 * 0.3;
                    }
                }
            }
            stack.record_post_state(&canvas, w, h);
        }

        assert_eq!(stack.undo_count(), 3);

        // Undo all 3
        stack.undo(&mut canvas, w, h);
        stack.undo(&mut canvas, w, h);
        stack.undo(&mut canvas, w, h);

        // Canvas should match original
        assert_eq!(canvas, original, "canvas should match original after undoing all strokes");
    }
}
