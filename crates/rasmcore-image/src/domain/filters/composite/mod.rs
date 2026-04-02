//! composite filters

mod blend;
pub use blend::*;
mod blend_if;
pub use blend_if::*;
mod mask_apply;
pub use mask_apply::*;
mod match_color;
pub use match_color::*;

#[cfg(test)]
mod tests;
