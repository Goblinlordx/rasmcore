//! alpha filters

mod add_alpha;
pub use add_alpha::*;
mod flatten;
pub use flatten::*;
mod premultiply;
pub use premultiply::*;
mod remove_alpha;
pub use remove_alpha::*;
mod unpremultiply;
pub use unpremultiply::*;
