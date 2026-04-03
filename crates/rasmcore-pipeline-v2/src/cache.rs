//! Spatial cache for f32 tile reuse.
//!
//! Stores computed f32 tile regions and reuses them when overlapping
//! requests arrive (common with neighborhood ops like blur/sharpen
//! that expand the input region beyond the output tile).

use std::collections::HashMap;
use crate::rect::Rect;
use crate::hash::ContentHash;

/// Cached tile region with f32 pixel data.
struct CachedRegion {
    rect: Rect,
    data: Vec<f32>,
    /// Content hash of the node output for invalidation.
    hash: ContentHash,
}

/// Spatial cache for f32 tile data.
///
/// Stores up to `budget_bytes` of cached tile regions. When the budget
/// is exceeded, the oldest entries are evicted (FIFO).
pub struct SpatialCache {
    /// Cached regions per node ID.
    entries: HashMap<u32, Vec<CachedRegion>>,
    /// Total bytes currently cached.
    current_bytes: usize,
    /// Maximum bytes allowed.
    budget_bytes: usize,
    /// Insertion order for FIFO eviction.
    order: Vec<(u32, usize)>, // (node_id, entry_index)
}

impl SpatialCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            current_bytes: 0,
            budget_bytes,
            order: Vec::new(),
        }
    }

    /// Try to find cached f32 data that covers the requested region.
    ///
    /// Returns the cached data cropped to the request, or None if not cached.
    pub fn query(&self, node_id: u32, request: Rect) -> Option<Vec<f32>> {
        let entries = self.entries.get(&node_id)?;
        for entry in entries {
            if entry.rect.contains(&request) {
                // Crop from cached region to requested region
                let w = entry.rect.width as usize;
                let rw = request.width as usize;
                let rh = request.height as usize;
                let dx = (request.x - entry.rect.x) as usize;
                let dy = (request.y - entry.rect.y) as usize;

                let mut result = Vec::with_capacity(rw * rh * 4);
                for row in 0..rh {
                    let src_off = ((dy + row) * w + dx) * 4;
                    result.extend_from_slice(&entry.data[src_off..src_off + rw * 4]);
                }
                return Some(result);
            }
        }
        None
    }

    /// Store f32 tile data in the cache.
    pub fn store(&mut self, node_id: u32, rect: Rect, data: Vec<f32>, hash: ContentHash) {
        let byte_size = data.len() * 4; // f32 = 4 bytes

        // Evict until we have room
        while self.current_bytes + byte_size > self.budget_bytes && !self.order.is_empty() {
            let (evict_node, evict_idx) = self.order.remove(0);
            if let Some(entries) = self.entries.get_mut(&evict_node) {
                if evict_idx < entries.len() {
                    let removed = entries.remove(evict_idx);
                    self.current_bytes -= removed.data.len() * 4;
                }
            }
        }

        let entries = self.entries.entry(node_id).or_default();
        let idx = entries.len();
        entries.push(CachedRegion { rect, data, hash });
        self.current_bytes += byte_size;
        self.order.push((node_id, idx));
    }

    /// Clear all cached data.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
        self.order.clear();
    }

    /// Current cache usage in bytes.
    pub fn usage_bytes(&self) -> usize {
        self.current_bytes
    }
}
