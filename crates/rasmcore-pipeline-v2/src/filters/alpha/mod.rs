//! Alpha channel and color matching filters — per-pixel operations.
//!
//! add_alpha, remove_alpha, flatten (composite over background),
//! match_color (histogram-based color transfer).

pub mod add_alpha;
pub mod flatten;
pub mod match_color;
pub mod remove_alpha;

pub use add_alpha::AddAlpha;
pub use flatten::Flatten;
pub use match_color::MatchColor;
pub use remove_alpha::RemoveAlpha;

#[cfg(test)]
mod tests {
    #[test]
    fn all_alpha_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["add_alpha", "remove_alpha", "flatten", "match_color"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }
}
