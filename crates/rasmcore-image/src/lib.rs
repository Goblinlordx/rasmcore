/// rasmcore-image: Image processing WASM component
///
/// Architecture: Ports/Adapters (Hexagonal)
/// - This file (lib.rs) is the ADAPTER layer — thin WIT binding glue
/// - Domain logic lives in domain/ — fully testable without WASM
/// - Domain defines its own error types; adapter translates to WIT errors

mod domain;

#[allow(warnings)]
mod bindings;

use bindings::exports::rasmcore::image::{decoder, encoder, filters, transform};
use bindings::rasmcore::core::{errors::RasmcoreError, types};

/// Adapter: converts domain errors to WIT rasmcore-error
fn to_wit_error(e: domain::error::ImageError) -> RasmcoreError {
    match e {
        domain::error::ImageError::InvalidInput(msg) => RasmcoreError::InvalidInput(msg),
        domain::error::ImageError::UnsupportedFormat(msg) => RasmcoreError::UnsupportedFormat(msg),
        domain::error::ImageError::NotImplemented => RasmcoreError::NotImplemented,
        domain::error::ImageError::ProcessingFailed(msg) => RasmcoreError::CodecError(msg),
        domain::error::ImageError::InvalidParameters(msg) => RasmcoreError::InvalidInput(msg),
    }
}

struct Component;

bindings::export!(Component with_types_in bindings);

impl decoder::Guest for Component {
    fn detect_format(_header: Vec<u8>) -> Option<String> {
        None // stub
    }

    fn decode(_data: Vec<u8>) -> Result<decoder::DecodedImage, RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn decode_as(
        _data: Vec<u8>,
        _target_format: types::PixelFormat,
    ) -> Result<decoder::DecodedImage, RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn supported_formats() -> Vec<String> {
        vec![]
    }
}

impl encoder::Guest for Component {
    fn encode(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _format: String,
        _quality: Option<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn supported_formats() -> Vec<String> {
        vec![]
    }
}

impl transform::Guest for Component {
    fn resize(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _width: u32,
        _height: u32,
        _filter: transform::ResizeFilter,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn crop(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _x: u32,
        _y: u32,
        _width: u32,
        _height: u32,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn rotate(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _degrees: transform::Rotation,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn flip(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _direction: transform::FlipDirection,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn convert_format(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _target: types::PixelFormat,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }
}

impl filters::Guest for Component {
    fn blur(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _radius: f32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn sharpen(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _amount: f32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn brightness(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _amount: f32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn contrast(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
        _amount: f32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }

    fn grayscale(
        _pixels: Vec<u8>,
        _info: types::ImageInfo,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        Err(to_wit_error(domain::error::ImageError::NotImplemented))
    }
}
