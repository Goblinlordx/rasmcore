//! Generic metadata container — flows through pipeline nodes alongside pixels.
//!
//! Each node receives upstream metadata and can read, modify, or pass it through.
//! Keys are string-namespaced by convention (e.g., "exif.Orientation", "icc_profile").
//! Include/exclude with glob patterns controls what gets embedded at write time.

use std::collections::HashMap;

/// A metadata value — supports common types found in image metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    String(String),
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
    Bool(bool),
}

impl MetadataValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            MetadataValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            MetadataValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            MetadataValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Generic metadata container with string-keyed values.
///
/// Keys use dot-separated namespacing by convention:
/// - `exif.Orientation`, `exif.Artist`, `exif.GPSLatitude`
/// - `iptc.Caption`, `iptc.Keywords`
/// - `xmp.Creator`, `xmp.Description`
/// - `icc_profile` (raw ICC profile bytes)
/// - `width`, `height`, `format` (image info)
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: HashMap<String, MetadataValue>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }


    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Set a value. Overwrites if key exists.
    pub fn set(&mut self, key: impl Into<String>, value: MetadataValue) {
        self.entries.insert(key.into(), value);
    }

    /// Remove a key. Returns the value if it existed.
    pub fn remove(&mut self, key: &str) -> Option<MetadataValue> {
        self.entries.remove(key)
    }

    /// Iterate over all keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|k| k.as_str())
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries as a reference.
    pub fn entries(&self) -> &HashMap<String, MetadataValue> {
        &self.entries
    }

    // ─── Typed Helpers ───────────────────────────────────────────────────

    /// Get ICC color profile bytes.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.get("icc_profile")?.as_bytes()
    }

    /// Get EXIF orientation (1-8).
    pub fn exif_orientation(&self) -> Option<u32> {
        self.get("exif.Orientation")?.as_int().map(|v| v as u32)
    }

    /// Get image width from metadata.
    pub fn width(&self) -> Option<u32> {
        self.get("width")?.as_int().map(|v| v as u32)
    }

    /// Get image height from metadata.
    pub fn height(&self) -> Option<u32> {
        self.get("height")?.as_int().map(|v| v as u32)
    }

    /// Get format string (e.g., "jpeg", "png").
    pub fn format(&self) -> Option<&str> {
        self.get("format")?.as_string()
    }

    // ─── Include / Exclude (Glob Filtering) ──────────────────────────────

    /// Return a new Metadata containing only keys matching any of the patterns.
    /// Patterns support `*` as wildcard (e.g., "exif.*", "exif.GPS*").
    pub fn include(&self, patterns: &[&str]) -> Metadata {
        let mut result = Metadata::new();
        for (key, value) in &self.entries {
            if patterns.iter().any(|p| glob_match(p, key)) {
                result.entries.insert(key.clone(), value.clone());
            }
        }
        result
    }

    /// Return a new Metadata with keys matching any of the patterns removed.
    pub fn exclude(&self, patterns: &[&str]) -> Metadata {
        let mut result = Metadata::new();
        for (key, value) in &self.entries {
            if !patterns.iter().any(|p| glob_match(p, key)) {
                result.entries.insert(key.clone(), value.clone());
            }
        }
        result
    }

    /// Serialize to JSON string (for WIT metadata-dump).
    pub fn to_json(&self) -> String {
        let mut entries = serde_json::Map::new();
        for (key, value) in &self.entries {
            let json_val = match value {
                MetadataValue::String(s) => serde_json::Value::String(s.clone()),
                MetadataValue::Int(i) => serde_json::json!(*i),
                MetadataValue::Float(f) => serde_json::json!(*f),
                MetadataValue::Bool(b) => serde_json::Value::Bool(*b),
                MetadataValue::Bytes(b) => {
                    // Base64 encode binary data for JSON
                    serde_json::Value::String(format!("base64:{}", base64_encode(b)))
                }
            };
            entries.insert(key.clone(), json_val);
        }
        serde_json::Value::Object(entries).to_string()
    }
}

/// Simple glob matching: `*` matches any sequence of characters.
///
/// - `"exif.*"` matches `"exif.Artist"`, `"exif.GPSLatitude"`
/// - `"exif.GPS*"` matches `"exif.GPSLatitude"`, `"exif.GPSLongitude"`
/// - `"exif.Artist"` matches only `"exif.Artist"` (exact)
/// - `"*"` matches everything
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let mut pi = pattern.chars().peekable();
    let mut ti = text.chars().peekable();

    while pi.peek().is_some() || ti.peek().is_some() {
        match pi.peek() {
            Some('*') => {
                pi.next();
                // * matches zero or more characters
                if pi.peek().is_none() {
                    return true; // trailing * matches everything
                }
                // Try matching rest of pattern at every position
                let remaining_pattern: String = pi.collect();
                let remaining_text: String = ti.collect();
                for i in 0..=remaining_text.len() {
                    if glob_match(&remaining_pattern, &remaining_text[i..]) {
                        return true;
                    }
                }
                return false;
            }
            Some(&pc) => match ti.next() {
                Some(tc) if tc == pc => {
                    pi.next();
                }
                _ => return false,
            },
            None => return ti.peek().is_none(),
        }
    }
    true
}

/// Simple base64 encoding (no external dependency).
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Metadata filter mode for write operations.
#[derive(Debug, Clone)]
pub enum MetadataFilter {
    /// Drop all metadata (default — safe, no PII leak).
    DropAll,
    /// Keep all metadata.
    KeepAll,
    /// Whitelist: only keys matching these glob patterns are kept.
    Include(Vec<String>),
    /// Blacklist: keys matching these glob patterns are removed.
    Exclude(Vec<String>),
}

impl Default for MetadataFilter {
    fn default() -> Self {
        MetadataFilter::DropAll
    }
}

impl MetadataFilter {
    /// Apply this filter to a metadata set.
    pub fn apply(&self, metadata: &Metadata) -> Metadata {
        match self {
            MetadataFilter::DropAll => Metadata::new(),
            MetadataFilter::KeepAll => metadata.clone(),
            MetadataFilter::Include(patterns) => {
                let refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
                metadata.include(&refs)
            }
            MetadataFilter::Exclude(patterns) => {
                let refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
                metadata.exclude(&refs)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_get() {
        let mut m = Metadata::new();
        m.set("exif.Artist", MetadataValue::String("Ben".into()));
        assert_eq!(m.get("exif.Artist").unwrap().as_string(), Some("Ben"));
    }

    #[test]
    fn remove() {
        let mut m = Metadata::new();
        m.set("key", MetadataValue::Int(42));
        assert!(m.remove("key").is_some());
        assert!(m.get("key").is_none());
    }

    #[test]
    fn typed_helpers() {
        let mut m = Metadata::new();
        m.set("icc_profile", MetadataValue::Bytes(vec![1, 2, 3]));
        m.set("exif.Orientation", MetadataValue::Int(6));
        m.set("width", MetadataValue::Int(4000));
        m.set("height", MetadataValue::Int(3000));
        m.set("format", MetadataValue::String("jpeg".into()));

        assert_eq!(m.icc_profile(), Some(&[1, 2, 3][..]));
        assert_eq!(m.exif_orientation(), Some(6));
        assert_eq!(m.width(), Some(4000));
        assert_eq!(m.height(), Some(3000));
        assert_eq!(m.format(), Some("jpeg"));
    }

    #[test]
    fn glob_exact() {
        assert!(glob_match("exif.Artist", "exif.Artist"));
        assert!(!glob_match("exif.Artist", "exif.Copyright"));
    }

    #[test]
    fn glob_star_suffix() {
        assert!(glob_match("exif.*", "exif.Artist"));
        assert!(glob_match("exif.*", "exif.GPSLatitude"));
        assert!(!glob_match("exif.*", "iptc.Caption"));
    }

    #[test]
    fn glob_star_prefix_match() {
        assert!(glob_match("exif.GPS*", "exif.GPSLatitude"));
        assert!(glob_match("exif.GPS*", "exif.GPSLongitude"));
        assert!(!glob_match("exif.GPS*", "exif.Artist"));
    }

    #[test]
    fn glob_star_everything() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("*", "exif.Artist"));
    }

    #[test]
    fn include_filter() {
        let mut m = Metadata::new();
        m.set("exif.Artist", MetadataValue::String("Ben".into()));
        m.set("exif.GPSLatitude", MetadataValue::Float(37.7));
        m.set("iptc.Caption", MetadataValue::String("Photo".into()));

        let filtered = m.include(&["exif.Artist", "iptc.*"]);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.get("exif.Artist").is_some());
        assert!(filtered.get("iptc.Caption").is_some());
        assert!(filtered.get("exif.GPSLatitude").is_none());
    }

    #[test]
    fn exclude_filter() {
        let mut m = Metadata::new();
        m.set("exif.Artist", MetadataValue::String("Ben".into()));
        m.set("exif.GPSLatitude", MetadataValue::Float(37.7));
        m.set("exif.GPSLongitude", MetadataValue::Float(-122.4));
        m.set("iptc.Caption", MetadataValue::String("Photo".into()));

        let filtered = m.exclude(&["exif.GPS*"]);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.get("exif.Artist").is_some());
        assert!(filtered.get("iptc.Caption").is_some());
        assert!(filtered.get("exif.GPSLatitude").is_none());
        assert!(filtered.get("exif.GPSLongitude").is_none());
    }

    #[test]
    fn metadata_filter_drop_all() {
        let mut m = Metadata::new();
        m.set("key", MetadataValue::Int(1));
        let result = MetadataFilter::DropAll.apply(&m);
        assert!(result.is_empty());
    }

    #[test]
    fn metadata_filter_keep_all() {
        let mut m = Metadata::new();
        m.set("key", MetadataValue::Int(1));
        let result = MetadataFilter::KeepAll.apply(&m);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn metadata_filter_include_patterns() {
        let mut m = Metadata::new();
        m.set("exif.Artist", MetadataValue::String("Ben".into()));
        m.set("exif.GPS", MetadataValue::Float(0.0));
        m.set("other", MetadataValue::Int(1));

        let filter = MetadataFilter::Include(vec!["exif.*".into()]);
        let result = filter.apply(&m);
        assert_eq!(result.len(), 2);
        assert!(result.get("other").is_none());
    }

    #[test]
    fn metadata_filter_exclude_patterns() {
        let mut m = Metadata::new();
        m.set("exif.Artist", MetadataValue::String("Ben".into()));
        m.set("exif.GPS", MetadataValue::Float(0.0));
        m.set("other", MetadataValue::Int(1));

        let filter = MetadataFilter::Exclude(vec!["exif.GPS*".into()]);
        let result = filter.apply(&m);
        assert_eq!(result.len(), 2);
        assert!(result.get("exif.GPS").is_none());
    }

    #[test]
    fn to_json_roundtrip() {
        let mut m = Metadata::new();
        m.set("exif.Artist", MetadataValue::String("Ben".into()));
        m.set("width", MetadataValue::Int(4000));
        let json = m.to_json();
        assert!(json.contains("Ben"));
        assert!(json.contains("4000"));
    }

    #[test]
    fn empty_metadata() {
        let m = Metadata::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert!(m.icc_profile().is_none());
        assert!(m.exif_orientation().is_none());
    }
}
