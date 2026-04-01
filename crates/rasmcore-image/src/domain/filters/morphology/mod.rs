//! morphology filters

mod dilate;
pub use dilate::*;
mod erode;
pub use erode::*;
mod morph_blackhat;
pub use morph_blackhat::*;
mod morph_close;
pub use morph_close::*;
mod morph_gradient;
pub use morph_gradient::*;
mod morph_open;
pub use morph_open::*;
mod morph_tophat;
pub use morph_tophat::*;

#[cfg(test)]
mod tests;
