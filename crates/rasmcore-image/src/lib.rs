// Allow proc macros to use ::rasmcore_image:: paths from within this crate
extern crate self as rasmcore_image;

/// rasmcore-image: Image processing WASM component
///
/// Architecture: Ports/Adapters (Hexagonal)
/// - Domain logic lives in domain/ — fully testable without WASM
/// - The WASM adapter (bindings + impl) is gated behind target_arch = "wasm32"
/// - Domain defines its own error types; adapter translates to WIT errors
pub mod domain;
