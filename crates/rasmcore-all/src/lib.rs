//! rasmcore-all — everything in one dependency.
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! rasmcore-all = "0.1"
//! ```
//!
//! This pulls in:
//! - `rasmcore-image` — pipeline infrastructure, types, registry
//! - `rasmcore-filters-std` — all standard filter categories
//! - `rasmcore-codecs-std` — all free codec support
//!
//! # Minimal builds
//!
//! For smaller binaries, depend on specific crates instead:
//!
//! ```toml
//! [dependencies]
//! rasmcore-image = "0.1"           # infrastructure only
//! rasmcore-filter-spatial = "0.1"  # just blur, sharpen, convolve, median
//! rasmcore-codec-png = "0.1"       # just PNG
//! rasmcore-codec-jpeg = "0.1"      # just JPEG
//! ```
//!
//! # Custom plugins
//!
//! Third parties can add their own filters/codecs:
//!
//! ```toml
//! [dependencies]
//! rasmcore-image = "0.1"
//! my-custom-filter = "0.1"  # uses #[register_filter]
//! ```

pub use rasmcore_codecs_std;
pub use rasmcore_filters_std;
pub use rasmcore_image;
