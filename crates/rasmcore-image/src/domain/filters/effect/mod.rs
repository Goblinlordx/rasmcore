//! effect filters

mod chromatic_aberration;
pub use chromatic_aberration::*;
mod chromatic_split;
pub use chromatic_split::*;
mod charcoal;
pub use charcoal::*;
mod emboss;
pub use emboss::*;
mod film_grain;
pub use film_grain::*;
mod gaussian_noise;
pub use gaussian_noise::*;
mod glitch;
pub use glitch::*;
mod halftone;
pub use halftone::*;
mod light_leak;
pub use light_leak::*;
mod mirror_kaleidoscope;
pub use mirror_kaleidoscope::*;
mod oil_paint;
pub use oil_paint::*;
mod pixelate;
pub use pixelate::*;
mod poisson_noise;
pub use poisson_noise::*;
mod salt_pepper_noise;
pub use salt_pepper_noise::*;
mod solarize;
pub use solarize::*;
mod uniform_noise;
pub use uniform_noise::*;

#[cfg(test)]
mod tests;
