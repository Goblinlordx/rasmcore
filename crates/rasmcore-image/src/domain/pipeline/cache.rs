//! Spatial cache with ref-counted region borrowing and sub-region reuse.
//!
//! Nodes request arbitrary rectangular regions. The cache:
//! 1. Checks if the region (or parts) is already cached
//! 2. Returns cached data with incremented ref count
//! 3. Reports missing sub-regions for the caller to compute
//! 4. Reclaims slots when ref count hits zero and memory budget is exceeded

use super::rect::Rect;

/// Unique identity for a cached region — scoped to a specific node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RegionKey {
    pub node_id: u32,
    pub rect: Rect,
}

/// Handle to a cached region. Lightweight, copyable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionHandle {
    pub(crate) index: usize,
    pub(crate) generation: u32,
}

/// A cached pixel region slot.
struct CacheSlot {
    key: Option<RegionKey>,
    pixels: Vec<u8>,
    rc: u32,
    generation: u32,
    bytes_per_pixel: u32,
}

/// Result of a spatial query against the cache.
pub struct CacheQuery {
    /// Existing cached regions that overlap the request.
    pub hits: Vec<RegionHandle>,
    /// Rectangles within the request that are NOT in cache and need computing.
    pub missing: Vec<Rect>,
    /// True if the entire request is covered by cache.
    pub fully_cached: bool,
}

/// Pipeline-owned spatial cache.
pub struct SpatialCache {
    slots: Vec<CacheSlot>,
    memory_budget: usize,
    memory_used: usize,
}

impl SpatialCache {
    pub fn new(memory_budget: usize) -> Self {
        Self {
            slots: Vec::new(),
            memory_budget,
            memory_used: 0,
        }
    }

    /// Query which parts of a requested region are cached for a given node.
    pub fn query(&self, node_id: u32, request: Rect) -> CacheQuery {
        let mut covered_rects = Vec::new();
        let mut hits = Vec::new();

        for (i, slot) in self.slots.iter().enumerate() {
            let Some(key) = &slot.key else { continue };
            if key.node_id != node_id {
                continue;
            }
            if key.rect.intersects(&request) {
                covered_rects.push(key.rect);
                hits.push(RegionHandle {
                    index: i,
                    generation: slot.generation,
                });
            }
        }

        let missing = request.difference_all(&covered_rects);
        let fully_cached = missing.is_empty();

        CacheQuery {
            hits,
            missing,
            fully_cached,
        }
    }

    /// Store a computed region in the cache. Returns a handle.
    pub fn store(
        &mut self,
        node_id: u32,
        rect: Rect,
        pixels: Vec<u8>,
        bytes_per_pixel: u32,
    ) -> RegionHandle {
        let byte_count = pixels.len();

        // Try to find a free slot (rc=0 with no key, or evict)
        let slot_idx = self.find_or_alloc_slot(byte_count);

        let slot = &mut self.slots[slot_idx];
        self.memory_used -= slot.pixels.len();
        slot.key = Some(RegionKey { node_id, rect });
        slot.pixels = pixels;
        slot.rc = 1;
        slot.generation += 1;
        slot.bytes_per_pixel = bytes_per_pixel;
        self.memory_used += byte_count;

        RegionHandle {
            index: slot_idx,
            generation: slot.generation,
        }
    }

    /// Increment ref count on a cached region (borrow it).
    pub fn acquire(&mut self, handle: RegionHandle) -> bool {
        if let Some(slot) = self.slots.get_mut(handle.index)
            && slot.generation == handle.generation
        {
            slot.rc += 1;
            return true;
        }
        false
    }

    /// Decrement ref count. Slot becomes reclaimable when rc hits 0.
    pub fn release(&mut self, handle: RegionHandle) {
        if let Some(slot) = self.slots.get_mut(handle.index) {
            assert_eq!(slot.generation, handle.generation, "stale region handle");
            assert!(slot.rc > 0, "double release");
            slot.rc -= 1;
        }
    }

    /// Read the pixels for a cached region.
    pub fn read(&self, handle: RegionHandle) -> &[u8] {
        let slot = &self.slots[handle.index];
        assert_eq!(slot.generation, handle.generation, "stale region handle");
        &slot.pixels
    }

    /// Get the rect for a cached region.
    pub fn rect(&self, handle: RegionHandle) -> Rect {
        let slot = &self.slots[handle.index];
        assert_eq!(slot.generation, handle.generation);
        slot.key.as_ref().expect("slot has no key").rect
    }

    /// Extract a sub-region from a cached region into a new buffer.
    /// Useful when a cached region is larger than what the caller needs.
    pub fn extract_subregion(&self, handle: RegionHandle, sub: Rect, bpp: u32) -> Vec<u8> {
        let slot = &self.slots[handle.index];
        assert_eq!(slot.generation, handle.generation);
        let parent = slot.key.as_ref().expect("slot has no key").rect;

        assert!(parent.contains(&sub), "sub-region not within cached region");

        let parent_stride = parent.width as usize * bpp as usize;
        let sub_stride = sub.width as usize * bpp as usize;
        let x_offset = (sub.x - parent.x) as usize * bpp as usize;
        let y_offset = (sub.y - parent.y) as usize;

        let mut result = Vec::with_capacity(sub.height as usize * sub_stride);
        for row in 0..sub.height as usize {
            let src_start = (y_offset + row) * parent_stride + x_offset;
            result.extend_from_slice(&slot.pixels[src_start..src_start + sub_stride]);
        }
        result
    }

    /// Current memory usage in bytes.
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// Number of active slots (rc > 0).
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.rc > 0).count()
    }

    fn find_or_alloc_slot(&mut self, needed_bytes: usize) -> usize {
        // First: look for an empty slot
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.key.is_none() && slot.rc == 0 {
                return i;
            }
        }

        // Second: evict a reclaimable slot (rc=0) if over budget
        if self.memory_used + needed_bytes > self.memory_budget {
            for (i, slot) in self.slots.iter_mut().enumerate() {
                if slot.rc == 0 && slot.key.is_some() {
                    self.memory_used -= slot.pixels.len();
                    slot.key = None;
                    slot.pixels.clear();
                    return i;
                }
            }
        }

        // Third: allocate a new slot
        self.slots.push(CacheSlot {
            key: None,
            pixels: Vec::new(),
            rc: 0,
            generation: 0,
            bytes_per_pixel: 0,
        });
        self.slots.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_read() {
        let mut cache = SpatialCache::new(1024 * 1024);
        let pixels = vec![1u8; 100];
        let handle = cache.store(0, Rect::new(0, 0, 10, 10), pixels.clone(), 1);
        assert_eq!(cache.read(handle), &pixels[..]);
    }

    #[test]
    fn query_hit() {
        let mut cache = SpatialCache::new(1024 * 1024);
        cache.store(0, Rect::new(0, 0, 100, 100), vec![0u8; 10000], 1);
        let q = cache.query(0, Rect::new(10, 10, 20, 20));
        assert!(q.fully_cached);
        assert!(q.missing.is_empty());
        assert_eq!(q.hits.len(), 1);
    }

    #[test]
    fn query_miss() {
        let mut cache = SpatialCache::new(1024 * 1024);
        cache.store(0, Rect::new(0, 0, 50, 50), vec![0u8; 2500], 1);
        let q = cache.query(0, Rect::new(60, 60, 20, 20));
        assert!(!q.fully_cached);
        assert_eq!(q.missing.len(), 1);
        assert_eq!(q.missing[0], Rect::new(60, 60, 20, 20));
    }

    #[test]
    fn query_partial_hit() {
        let mut cache = SpatialCache::new(1024 * 1024);
        cache.store(0, Rect::new(0, 0, 50, 50), vec![0u8; 2500], 1);
        // Request overlaps cached region: needs rows 40-70
        let q = cache.query(0, Rect::new(0, 40, 50, 30));
        assert!(!q.fully_cached);
        assert!(!q.missing.is_empty());
        // Missing should be the uncovered part (rows 50-70)
        let missing_area: u64 = q.missing.iter().map(|r| r.area()).sum();
        assert_eq!(missing_area, 50 * 20); // 50 wide, 20 rows uncovered
    }

    #[test]
    fn query_different_nodes() {
        let mut cache = SpatialCache::new(1024 * 1024);
        cache.store(0, Rect::new(0, 0, 100, 100), vec![0u8; 10000], 1);
        // Query for node 1 should miss even though node 0 has data
        let q = cache.query(1, Rect::new(0, 0, 50, 50));
        assert!(!q.fully_cached);
        assert_eq!(q.missing.len(), 1);
    }

    #[test]
    fn ref_counting() {
        let mut cache = SpatialCache::new(1024 * 1024);
        let h = cache.store(0, Rect::new(0, 0, 10, 10), vec![0u8; 100], 1);
        // rc is 1 after store
        cache.acquire(h);
        // rc is 2
        cache.release(h);
        // rc is 1
        cache.release(h);
        // rc is 0 — reclaimable
        assert_eq!(cache.active_count(), 0);
    }

    #[test]
    fn eviction_on_budget() {
        let mut cache = SpatialCache::new(200);
        let h1 = cache.store(0, Rect::new(0, 0, 10, 10), vec![0u8; 100], 1);
        cache.release(h1); // rc=0, reclaimable
        // This store should evict h1
        let _h2 = cache.store(0, Rect::new(10, 10, 15, 15), vec![0u8; 150], 1);
        assert!(cache.memory_used() <= 200);
    }

    #[test]
    fn extract_subregion() {
        let mut cache = SpatialCache::new(1024 * 1024);
        // 4x4 image, 1 bpp, values are row*4+col
        let pixels: Vec<u8> = (0..16).collect();
        let h = cache.store(0, Rect::new(0, 0, 4, 4), pixels, 1);
        // Extract 2x2 subregion at (1,1)
        let sub = cache.extract_subregion(h, Rect::new(1, 1, 2, 2), 1);
        assert_eq!(sub, vec![5, 6, 9, 10]);
    }
}
