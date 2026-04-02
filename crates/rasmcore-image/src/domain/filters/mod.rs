//! Image filters — organized by category, one filter per file.
//!
//! Shared helpers, enums, constants, and structs live in `common`.
//! ConfigParams structs live in their respective filter files.

pub mod common;

pub mod adjustment;
pub mod advanced;
pub mod alpha;
pub mod analysis;
pub mod color;
pub mod composite;
pub mod distortion;
pub mod draw;
pub mod edge;
pub mod effect;
pub mod enhancement;
pub mod evaluate;
pub mod generator;
pub mod grading;
pub mod mask;
pub mod morphology;
pub mod spatial;
pub mod threshold;
pub mod tonemapping;
pub mod tool;
pub mod transform;

pub mod kernels {
    /// 3x3 emboss kernel.
    pub const EMBOSS: [f32; 9] = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
    /// 3x3 edge-enhance kernel.
    pub const EDGE_ENHANCE: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
    /// 3x3 sharpen kernel.
    pub const SHARPEN_3X3: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
    /// 3x3 box blur kernel.
    pub const BOX_BLUR_3X3: [f32; 9] = [1.0; 9];
}

// Re-export everything for backward compatibility
pub use common::*;
pub use adjustment::*;
pub use advanced::*;
pub use alpha::*;
pub use analysis::*;
pub use color::*;
pub use composite::*;
pub use distortion::*;
pub use draw::*;
pub use edge::*;
pub use effect::*;
pub use enhancement::*;
pub use evaluate::*;
pub use generator::*;
pub use grading::*;
pub use mask::*;
pub use morphology::*;
pub use spatial::*;
pub use threshold::*;
pub use tonemapping::*;
pub use tool::*;
pub use transform::*;
