//! Spatial cache for f32 tile reuse.
//!
//! Stores computed f32 tile regions and reuses them when overlapping
//! requests arrive (common with neighborhood ops like blur/sharpen
//! that expand the input region beyond the output tile).
//!
//! Uses LRU eviction bounded by both entry count and byte budget.

use std::collections::HashMap;
use crate::rect::Rect;
use crate::hash::ContentHash;

/// Cached tile region with f32 pixel data.
struct CachedRegion {
    rect: Rect,
    data: Vec<f32>,
    /// Content hash of the node output for invalidation.
    _hash: ContentHash,
    /// LRU generation — higher = more recently used.
    generation: u64,
}

/// Spatial cache for f32 tile data.
///
/// Stores tile regions with LRU eviction bounded by:
/// 1. Entry count: `max_entries` (default: 256, updated via `set_node_count`)
/// 2. Byte budget: `budget_bytes` (default: 16 MB)
///
/// Whichever limit is hit first triggers eviction of the oldest entries.
pub struct SpatialCache {
    /// Cached regions per node ID.
    entries: HashMap<u32, Vec<CachedRegion>>,
    /// Total bytes currently cached.
    current_bytes: usize,
    /// Maximum bytes allowed.
    budget_bytes: usize,
    /// Maximum entry count. Updated by set_node_count().
    max_entries: usize,
    /// Current total entry count.
    entry_count: usize,
    /// Monotonically increasing generation counter for LRU tracking.
    generation: u64,
}

impl SpatialCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            current_bytes: 0,
            budget_bytes,
            max_entries: 256, // default, updated by set_node_count
            entry_count: 0,
            generation: 0,
        }
    }

    /// Update the max entry count based on graph node count.
    /// Formula: n * ceil(log2(n)), minimum 16.
    pub fn set_node_count(&mut self, n: usize) {
        let n = n.max(1);
        let log2 = (usize::BITS - n.leading_zeros()) as usize; // ceil(log2(n))
        self.max_entries = (n * log2).max(16);
    }

    /// Try to find cached f32 data that covers the requested region.
    ///
    /// Returns the cached data cropped to the request, or None if not cached.
    /// Updates LRU generation on hit.
    pub fn query(&mut self, node_id: u32, request: Rect) -> Option<Vec<f32>> {
        self.generation += 1;
        let current_gen = self.generation;

        let entries = self.entries.get_mut(&node_id)?;
        for entry in entries.iter_mut() {
            if entry.rect.contains(&request) {
                // Mark as recently used
                entry.generation = current_gen;

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
        self.generation += 1;
        let current_gen = self.generation;

        // Evict by entry count limit
        while self.entry_count >= self.max_entries {
            if !self.evict_lru() {
                break;
            }
        }

        // Evict by byte budget
        while self.current_bytes + byte_size > self.budget_bytes {
            if !self.evict_lru() {
                break;
            }
        }

        let entries = self.entries.entry(node_id).or_default();
        entries.push(CachedRegion { rect, data, _hash: hash, generation: current_gen });
        self.current_bytes += byte_size;
        self.entry_count += 1;
    }

    /// Evict the least recently used entry across all nodes.
    /// Returns true if an entry was evicted, false if cache is empty.
    fn evict_lru(&mut self) -> bool {
        // Find the entry with the lowest generation across all nodes
        let mut oldest_node = 0u32;
        let mut oldest_idx = 0usize;
        let mut oldest_gen = u64::MAX;

        for (&node_id, entries) in &self.entries {
            for (idx, entry) in entries.iter().enumerate() {
                if entry.generation < oldest_gen {
                    oldest_gen = entry.generation;
                    oldest_node = node_id;
                    oldest_idx = idx;
                }
            }
        }

        if oldest_gen == u64::MAX {
            return false; // no entries
        }

        if let Some(entries) = self.entries.get_mut(&oldest_node) {
            let removed = entries.swap_remove(oldest_idx);
            self.current_bytes -= removed.data.len() * 4;
            self.entry_count -= 1;
            if entries.is_empty() {
                self.entries.remove(&oldest_node);
            }
        }
        true
    }

    /// Clear all cached data.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
        self.entry_count = 0;
        // Don't reset generation — monotonic counter survives clear
    }

    /// Invalidate all cached entries for a specific node.
    pub fn invalidate(&mut self, node_id: u32) {
        if let Some(entries) = self.entries.remove(&node_id) {
            for entry in &entries {
                self.current_bytes -= entry.data.len() * 4;
            }
            self.entry_count -= entries.len();
        }
    }

    /// Current cache usage in bytes.
    pub fn usage_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Budget capacity in bytes.
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Current number of cached entries.
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Maximum entries allowed.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
}

// ─── Spatial Cache Pool ────────────────────────────────────────────────────

use std::cell::RefCell;

/// Pool of reusable SpatialCache instances.
///
/// Avoids allocating HashMap/Vec on every pipeline creation. The pool
/// auto-sizes to the peak concurrent demand — if 4 pipelines run
/// concurrently, the pool grows to hold 4 caches. Released caches
/// are cleared and returned to the pool.
pub struct SpatialCachePool {
    available: Vec<SpatialCache>,
    budget_bytes: usize,
    /// Maximum idle caches to keep. Excess are dropped on release.
    max_idle: usize,
}

impl SpatialCachePool {
    /// Create a pool that produces caches with the given budget.
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            available: Vec::new(),
            budget_bytes,
            max_idle: 2,
        }
    }

    /// Acquire a cache from the pool (or create a new one).
    /// The cache is cleared but its internal allocations are reused.
    pub fn acquire(&mut self) -> SpatialCache {
        if let Some(mut cache) = self.available.pop() {
            cache.clear();
            cache
        } else {
            SpatialCache::new(self.budget_bytes)
        }
    }

    /// Return a cache to the pool for reuse.
    /// Drops the cache if pool is already at max_idle.
    pub fn release(&mut self, cache: SpatialCache) {
        if self.available.len() >= self.max_idle {
            return; // drop — pool is full
        }
        self.available.push(cache);
    }

    /// Number of caches currently in the pool (idle).
    pub fn idle_count(&self) -> usize {
        self.available.len()
    }
}

thread_local! {
    /// Global thread-local spatial cache pool.
    /// Default budget: 16 MB (matches Graph::new default).
    static CACHE_POOL: RefCell<SpatialCachePool> = RefCell::new(
        SpatialCachePool::new(16 * 1024 * 1024)
    );
}

/// Acquire a spatial cache from the thread-local pool.
pub fn acquire_pooled_cache() -> SpatialCache {
    CACHE_POOL.with(|pool| pool.borrow_mut().acquire())
}

/// Return a spatial cache to the thread-local pool.
pub fn release_pooled_cache(cache: SpatialCache) {
    CACHE_POOL.with(|pool| pool.borrow_mut().release(cache));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::ZERO_HASH;

    fn make_data(n: usize) -> Vec<f32> {
        vec![0.5f32; n * 4] // n pixels, 4 channels each
    }

    #[test]
    fn basic_store_and_query() {
        let mut cache = SpatialCache::new(1024 * 1024);
        let rect = Rect::new(0, 0, 4, 4);
        let data = make_data(16); // 4x4
        cache.store(0, rect, data.clone(), ZERO_HASH);
        let result = cache.query(0, rect);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), data);
    }

    #[test]
    fn query_subregion() {
        let mut cache = SpatialCache::new(1024 * 1024);
        let rect = Rect::new(0, 0, 4, 4);
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect(); // 4x4x4
        cache.store(0, rect, data, ZERO_HASH);
        // Query a 2x2 sub-region at (1,1)
        let sub = cache.query(0, Rect::new(1, 1, 2, 2));
        assert!(sub.is_some());
        let sub = sub.unwrap();
        assert_eq!(sub.len(), 2 * 2 * 4); // 2x2 pixels, 4 channels
    }

    #[test]
    fn entry_count_eviction() {
        let mut cache = SpatialCache::new(1024 * 1024); // large budget
        cache.set_node_count(2); // max_entries = 2 * ceil(log2(2)) = 2 * 2 = 4, min 16
        // min 16, so store 17 entries
        let max = cache.max_entries();
        for i in 0..(max + 1) as u32 {
            cache.store(i, Rect::new(0, 0, 1, 1), make_data(1), ZERO_HASH);
        }
        // Should have evicted the oldest
        assert_eq!(cache.entry_count(), max);
        // Oldest (node 0) should be evicted
        assert!(cache.query(0, Rect::new(0, 0, 1, 1)).is_none());
        // Newest should still be present
        assert!(cache.query(max as u32, Rect::new(0, 0, 1, 1)).is_some());
    }

    #[test]
    fn byte_budget_eviction() {
        // Budget for exactly 2 entries of 4 pixels each (4*4*4 = 64 bytes each)
        let mut cache = SpatialCache::new(128);
        cache.set_node_count(100); // high entry limit so byte budget is the trigger
        cache.store(0, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH); // 64 bytes
        cache.store(1, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH); // 64 bytes — at budget
        cache.store(2, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH); // would exceed → evicts node 0
        assert!(cache.query(0, Rect::new(0, 0, 2, 2)).is_none());
        assert!(cache.query(1, Rect::new(0, 0, 2, 2)).is_some());
        assert!(cache.query(2, Rect::new(0, 0, 2, 2)).is_some());
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut cache = SpatialCache::new(128);
        cache.set_node_count(100);
        cache.store(0, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH); // gen 1
        cache.store(1, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH); // gen 2
        // Touch node 0 — makes it more recent
        let _ = cache.query(0, Rect::new(0, 0, 2, 2)); // gen 3
        // Add node 2 — should evict node 1 (oldest at gen 2)
        cache.store(2, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH);
        assert!(cache.query(0, Rect::new(0, 0, 2, 2)).is_some()); // kept (was touched)
        assert!(cache.query(1, Rect::new(0, 0, 2, 2)).is_none()); // evicted
        assert!(cache.query(2, Rect::new(0, 0, 2, 2)).is_some()); // just added
    }

    #[test]
    fn invalidate_node() {
        let mut cache = SpatialCache::new(1024 * 1024);
        cache.store(0, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH);
        cache.store(1, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH);
        assert_eq!(cache.entry_count(), 2);
        cache.invalidate(0);
        assert_eq!(cache.entry_count(), 1);
        assert!(cache.query(0, Rect::new(0, 0, 2, 2)).is_none());
        assert!(cache.query(1, Rect::new(0, 0, 2, 2)).is_some());
    }

    #[test]
    fn no_panic_on_repeated_evict_cycles() {
        let mut cache = SpatialCache::new(64); // tiny budget
        for i in 0..100u32 {
            cache.store(i, Rect::new(0, 0, 2, 2), make_data(4), ZERO_HASH);
        }
        // Should not panic and should have reasonable state
        assert!(cache.entry_count() <= 2); // budget fits at most 1
        assert!(cache.usage_bytes() <= 64);
    }

    #[test]
    fn clear_resets_state() {
        let mut cache = SpatialCache::new(1024 * 1024);
        cache.store(0, Rect::new(0, 0, 4, 4), make_data(16), ZERO_HASH);
        assert_eq!(cache.entry_count(), 1);
        cache.clear();
        assert_eq!(cache.entry_count(), 0);
        assert_eq!(cache.usage_bytes(), 0);
        assert!(cache.query(0, Rect::new(0, 0, 4, 4)).is_none());
    }
}
