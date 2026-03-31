//! Content-addressed layer cache for cross-pipeline result reuse.
//!
//! The LayerCache lives outside any single pipeline's lifetime. Each node's
//! output is keyed by a content hash that encodes the full computation lineage:
//! hash(upstream_hash || operation_name || param_bytes).
//!
//! Pipelines check the cache before execution and push results after completion.
//! The SpatialCache is NOT modified — LayerCache operates at the graph level.

use std::collections::HashMap;

/// Content hash — 32-byte blake3 digest encoding the full computation lineage.
pub type ContentHash = [u8; 32];

/// A zero hash used as the initial upstream for source nodes.
pub const ZERO_HASH: ContentHash = [0u8; 32];

/// Compute a content hash from an upstream hash, operation name, and parameters.
///
/// hash = blake3(upstream_hash || op_name || param_bytes)
pub fn compute_hash(upstream: &ContentHash, op: &str, params: &[u8]) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(upstream);
    hasher.update(op.as_bytes());
    hasher.update(params);
    *hasher.finalize().as_bytes()
}

/// Compute a source hash from input data (first 4KB + length for speed).
pub fn compute_source_hash(data: &[u8]) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"source");
    hasher.update(&(data.len() as u64).to_le_bytes());
    // Hash first 4KB + last 4KB for speed on large inputs
    let prefix = &data[..data.len().min(4096)];
    hasher.update(prefix);
    if data.len() > 4096 {
        let suffix_start = data.len().saturating_sub(4096);
        hasher.update(&data[suffix_start..]);
    }
    *hasher.finalize().as_bytes()
}

/// A cached layer entry: pixels + metadata.
struct CachedLayer {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    bpp: u32,
    referenced: bool,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub entries: u32,
    pub hits: u64,
    pub misses: u64,
    pub size_bytes: u64,
}

/// Default maximum cap: 1 GB. Leaves headroom in 4 GB WASM address space.
const DEFAULT_MAX_BUDGET: usize = 1024 * 1024 * 1024;

/// Content-addressed layer cache that persists across pipeline lifetimes.
///
/// Starts at the initial budget and auto-grows (doubling) up to `max_budget`
/// when a store would require evicting referenced entries. This avoids
/// thrashing on large images or long chains while keeping a safe cap.
pub struct LayerCache {
    entries: HashMap<ContentHash, CachedLayer>,
    memory_budget: usize,
    max_budget: usize,
    memory_used: usize,
    stats: CacheStats,
}

impl LayerCache {
    /// Create a new layer cache with the given initial memory budget in bytes.
    /// Auto-grows up to 1 GB when needed.
    pub fn new(memory_budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            memory_budget,
            max_budget: DEFAULT_MAX_BUDGET,
            memory_used: 0,
            stats: CacheStats::default(),
        }
    }

    /// Create with explicit initial and maximum budgets.
    pub fn with_max_budget(initial_budget: usize, max_budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            memory_budget: initial_budget,
            max_budget,
            memory_used: 0,
            stats: CacheStats::default(),
        }
    }

    /// Check if a hash is cached. Returns pixels if found, marks as referenced.
    pub fn get(&mut self, hash: &ContentHash) -> Option<(&[u8], u32, u32, u32)> {
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.referenced = true;
            self.stats.hits += 1;
            Some((&entry.pixels, entry.width, entry.height, entry.bpp))
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Store a layer's output in the cache.
    pub fn store(
        &mut self,
        hash: ContentHash,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        bpp: u32,
    ) {
        let size = pixels.len();

        // Evict unreferenced entries first
        while self.memory_used + size > self.memory_budget && !self.entries.is_empty() {
            let evict_key = self
                .entries
                .iter()
                .find(|(_, v)| !v.referenced)
                .map(|(k, _)| *k);
            if let Some(key) = evict_key {
                if let Some(removed) = self.entries.remove(&key) {
                    self.memory_used -= removed.pixels.len();
                }
            } else {
                // All entries referenced — try to grow instead of dropping
                if self.memory_budget < self.max_budget {
                    let new_budget = (self.memory_budget * 2).min(self.max_budget);
                    // Ensure growth is at least enough for this store
                    let needed = self.memory_used + size;
                    self.memory_budget = new_budget.max(needed).min(self.max_budget);
                }
                break;
            }
        }

        // If still doesn't fit after eviction, grow if under cap
        if self.memory_used + size > self.memory_budget && self.memory_budget < self.max_budget {
            let needed = self.memory_used + size;
            self.memory_budget = needed.min(self.max_budget);
        }

        // Store if fits (after possible growth)
        if self.memory_used + size <= self.memory_budget || self.entries.is_empty() {
            self.memory_used += size;
            self.entries.insert(
                hash,
                CachedLayer {
                    pixels,
                    width,
                    height,
                    bpp,
                    referenced: true,
                },
            );
        }
    }

    /// Mark a hash as referenced (even if it was a cache hit — still "used").
    pub fn mark_referenced(&mut self, hash: &ContentHash) {
        if let Some(entry) = self.entries.get_mut(hash) {
            entry.referenced = true;
        }
    }

    /// Reset all referenced flags (call at start of new pipeline run).
    pub fn reset_references(&mut self) {
        for entry in self.entries.values_mut() {
            entry.referenced = false;
        }
    }

    /// Remove all unreferenced entries. Call after pipeline completion.
    pub fn cleanup_unreferenced(&mut self) {
        self.entries.retain(|_, v| {
            if v.referenced {
                true
            } else {
                self.memory_used -= v.pixels.len();
                false
            }
        });
        self.stats.entries = self.entries.len() as u32;
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.memory_used = 0;
        self.stats = CacheStats::default();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len() as u32,
            size_bytes: self.memory_used as u64,
            ..self.stats
        }
    }

    /// Check if a hash exists without marking it referenced.
    pub fn contains(&self, hash: &ContentHash) -> bool {
        self.entries.contains_key(hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_chain_deterministic() {
        let src = compute_source_hash(b"test image data");
        let h1 = compute_hash(&src, "blur", &2.0f32.to_le_bytes());
        let h2 = compute_hash(&src, "blur", &2.0f32.to_le_bytes());
        assert_eq!(h1, h2, "same inputs must produce same hash");
    }

    #[test]
    fn hash_chain_different_params() {
        let src = compute_source_hash(b"test image data");
        let h1 = compute_hash(&src, "blur", &2.0f32.to_le_bytes());
        let h2 = compute_hash(&src, "blur", &3.0f32.to_le_bytes());
        assert_ne!(h1, h2, "different params must produce different hash");
    }

    #[test]
    fn hash_chain_upstream_propagates() {
        let src1 = compute_source_hash(b"image A");
        let src2 = compute_source_hash(b"image B");
        let h1 = compute_hash(&src1, "blur", &2.0f32.to_le_bytes());
        let h2 = compute_hash(&src2, "blur", &2.0f32.to_le_bytes());
        assert_ne!(h1, h2, "different upstream must produce different hash");
    }

    #[test]
    fn store_and_retrieve() {
        let mut cache = LayerCache::new(1024 * 1024);
        let hash = compute_source_hash(b"test");
        cache.store(hash, vec![1, 2, 3, 4], 2, 2, 1);
        let result = cache.get(&hash);
        assert!(result.is_some());
        let (pixels, w, h, bpp) = result.unwrap();
        assert_eq!(pixels, &[1, 2, 3, 4]);
        assert_eq!((w, h, bpp), (2, 2, 1));
    }

    #[test]
    fn miss_returns_none() {
        let mut cache = LayerCache::new(1024);
        let hash = compute_source_hash(b"not stored");
        assert!(cache.get(&hash).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn cleanup_removes_unreferenced() {
        let mut cache = LayerCache::new(1024 * 1024);
        let h1 = compute_source_hash(b"keep");
        let h2 = compute_source_hash(b"remove");
        cache.store(h1, vec![1; 100], 10, 10, 1);
        cache.store(h2, vec![2; 100], 10, 10, 1);

        // Reset and only reference h1
        cache.reset_references();
        cache.mark_referenced(&h1);
        cache.cleanup_unreferenced();

        assert!(cache.contains(&h1));
        assert!(!cache.contains(&h2));
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn memory_budget_enforced() {
        let mut cache = LayerCache::new(200); // tight budget
        let h1 = compute_source_hash(b"first");
        let h2 = compute_source_hash(b"second");
        cache.store(h1, vec![0; 100], 10, 10, 1);
        cache.store(h2, vec![0; 100], 10, 10, 1);

        // Both fit in 200 bytes
        assert!(cache.contains(&h1));
        assert!(cache.contains(&h2));

        // Third should evict unreferenced
        cache.reset_references();
        cache.mark_referenced(&h2); // only h2 referenced
        let h3 = compute_source_hash(b"third");
        cache.store(h3, vec![0; 100], 10, 10, 1);

        // h1 should be evicted (unreferenced), h2 and h3 should remain
        assert!(!cache.contains(&h1));
        assert!(cache.contains(&h2));
        assert!(cache.contains(&h3));
    }

    #[test]
    fn clear_removes_everything() {
        let mut cache = LayerCache::new(1024);
        cache.store(compute_source_hash(b"a"), vec![1], 1, 1, 1);
        cache.store(compute_source_hash(b"b"), vec![2], 1, 1, 1);
        cache.clear();
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().size_bytes, 0);
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let mut cache = LayerCache::new(1024);
        let hash = compute_source_hash(b"test");
        cache.store(hash, vec![1, 2, 3], 1, 1, 3);

        cache.get(&hash); // hit
        cache.get(&hash); // hit
        cache.get(&compute_source_hash(b"miss")); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }
}
