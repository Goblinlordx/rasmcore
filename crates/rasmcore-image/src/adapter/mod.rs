//! WASM adapter layer — thin WIT binding glue.
//! Only compiled for wasm32 targets.

mod pipeline_adapter;

use crate::bindings;
use crate::bindings::exports::rasmcore::image::{decoder, encoder, filters, pipeline, transform};
use crate::bindings::rasmcore::core::{errors::RasmcoreError, types};

use crate::domain;

fn to_wit_error(e: domain::error::ImageError) -> RasmcoreError {
    match e {
        domain::error::ImageError::InvalidInput(msg) => RasmcoreError::InvalidInput(msg),
        domain::error::ImageError::UnsupportedFormat(msg) => RasmcoreError::UnsupportedFormat(msg),
        domain::error::ImageError::NotImplemented => RasmcoreError::NotImplemented,
        domain::error::ImageError::ProcessingFailed(msg) => RasmcoreError::CodecError(msg),
        domain::error::ImageError::InvalidParameters(msg) => RasmcoreError::InvalidInput(msg),
    }
}

fn to_domain_pixel_format(f: types::PixelFormat) -> domain::types::PixelFormat {
    match f {
        types::PixelFormat::Rgb8 => domain::types::PixelFormat::Rgb8,
        types::PixelFormat::Rgba8 => domain::types::PixelFormat::Rgba8,
        types::PixelFormat::Bgr8 => domain::types::PixelFormat::Bgr8,
        types::PixelFormat::Bgra8 => domain::types::PixelFormat::Bgra8,
        types::PixelFormat::Gray8 => domain::types::PixelFormat::Gray8,
        types::PixelFormat::Gray16 => domain::types::PixelFormat::Gray16,
        types::PixelFormat::Yuv420p => domain::types::PixelFormat::Yuv420p,
        types::PixelFormat::Yuv422p => domain::types::PixelFormat::Yuv422p,
        types::PixelFormat::Yuv444p => domain::types::PixelFormat::Yuv444p,
        types::PixelFormat::Nv12 => domain::types::PixelFormat::Nv12,
    }
}

fn to_wit_pixel_format(f: domain::types::PixelFormat) -> types::PixelFormat {
    match f {
        domain::types::PixelFormat::Rgb8 => types::PixelFormat::Rgb8,
        domain::types::PixelFormat::Rgba8 => types::PixelFormat::Rgba8,
        domain::types::PixelFormat::Bgr8 => types::PixelFormat::Bgr8,
        domain::types::PixelFormat::Bgra8 => types::PixelFormat::Bgra8,
        domain::types::PixelFormat::Gray8 => types::PixelFormat::Gray8,
        domain::types::PixelFormat::Gray16 => types::PixelFormat::Gray16,
        domain::types::PixelFormat::Yuv420p => types::PixelFormat::Yuv420p,
        domain::types::PixelFormat::Yuv422p => types::PixelFormat::Yuv422p,
        domain::types::PixelFormat::Yuv444p => types::PixelFormat::Yuv444p,
        domain::types::PixelFormat::Nv12 => types::PixelFormat::Nv12,
    }
}

fn to_wit_color_space(c: domain::types::ColorSpace) -> types::ColorSpace {
    match c {
        domain::types::ColorSpace::Srgb => types::ColorSpace::Srgb,
        domain::types::ColorSpace::LinearSrgb => types::ColorSpace::LinearSrgb,
        domain::types::ColorSpace::DisplayP3 => types::ColorSpace::DisplayP3,
        domain::types::ColorSpace::Bt709 => types::ColorSpace::Bt709,
        domain::types::ColorSpace::Bt2020 => types::ColorSpace::Bt2020,
    }
}

fn to_wit_image_info(info: &domain::types::ImageInfo) -> types::ImageInfo {
    types::ImageInfo {
        width: info.width,
        height: info.height,
        format: to_wit_pixel_format(info.format),
        color_space: to_wit_color_space(info.color_space),
    }
}

fn to_domain_image_info(info: &types::ImageInfo) -> domain::types::ImageInfo {
    domain::types::ImageInfo {
        width: info.width,
        height: info.height,
        format: to_domain_pixel_format(info.format),
        color_space: match info.color_space {
            types::ColorSpace::Srgb => domain::types::ColorSpace::Srgb,
            types::ColorSpace::LinearSrgb => domain::types::ColorSpace::LinearSrgb,
            types::ColorSpace::DisplayP3 => domain::types::ColorSpace::DisplayP3,
            types::ColorSpace::Bt709 => domain::types::ColorSpace::Bt709,
            types::ColorSpace::Bt2020 => domain::types::ColorSpace::Bt2020,
        },
    }
}

fn to_domain_png_filter(f: Option<encoder::PngFilterType>) -> domain::encoder::png::PngFilterType {
    match f {
        None => domain::encoder::png::PngFilterType::Adaptive,
        Some(encoder::PngFilterType::NoFilter) => domain::encoder::png::PngFilterType::NoFilter,
        Some(encoder::PngFilterType::Sub) => domain::encoder::png::PngFilterType::Sub,
        Some(encoder::PngFilterType::Up) => domain::encoder::png::PngFilterType::Up,
        Some(encoder::PngFilterType::Avg) => domain::encoder::png::PngFilterType::Avg,
        Some(encoder::PngFilterType::Paeth) => domain::encoder::png::PngFilterType::Paeth,
        Some(encoder::PngFilterType::Adaptive) => domain::encoder::png::PngFilterType::Adaptive,
    }
}

struct Component;

bindings::export!(Component with_types_in bindings);

impl pipeline::Guest for Component {
    type ImagePipeline = pipeline_adapter::PipelineResource;

    fn detect_format(header: Vec<u8>) -> Option<String> {
        domain::decoder::detect_format(&header)
    }

    fn supported_read_formats() -> Vec<String> {
        domain::decoder::supported_formats()
    }

    fn supported_write_formats() -> Vec<String> {
        domain::encoder::supported_formats()
    }
}

impl decoder::Guest for Component {
    fn detect_format(header: Vec<u8>) -> Option<String> {
        domain::decoder::detect_format(&header)
    }

    fn decode(data: Vec<u8>) -> Result<decoder::DecodedImage, RasmcoreError> {
        let result = domain::decoder::decode(&data).map_err(to_wit_error)?;
        Ok(decoder::DecodedImage {
            pixels: result.pixels,
            info: to_wit_image_info(&result.info),
        })
    }

    fn decode_as(
        data: Vec<u8>,
        target_format: types::PixelFormat,
    ) -> Result<decoder::DecodedImage, RasmcoreError> {
        let fmt = to_domain_pixel_format(target_format);
        let result = domain::decoder::decode_as(&data, fmt).map_err(to_wit_error)?;
        Ok(decoder::DecodedImage {
            pixels: result.pixels,
            info: to_wit_image_info(&result.info),
        })
    }

    fn supported_formats() -> Vec<String> {
        domain::decoder::supported_formats()
    }
}

impl encoder::Guest for Component {
    fn encode_jpeg(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::JpegEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let img = domain::encoder::pixels_to_dynamic_image(&pixels, &domain_info)
            .map_err(to_wit_error)?;
        let domain_config = domain::encoder::jpeg::JpegEncodeConfig {
            quality: config.quality.unwrap_or(85),
            progressive: config.progressive.unwrap_or(false),
        };
        domain::encoder::jpeg::encode(&img, &domain_info, &domain_config).map_err(to_wit_error)
    }

    fn encode_png(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::PngEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let img = domain::encoder::pixels_to_dynamic_image(&pixels, &domain_info)
            .map_err(to_wit_error)?;
        let domain_config = domain::encoder::png::PngEncodeConfig {
            compression_level: config.compression_level.unwrap_or(6),
            filter_type: to_domain_png_filter(config.filter_type),
        };
        domain::encoder::png::encode(&img, &domain_info, &domain_config).map_err(to_wit_error)
    }

    fn encode_webp(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::WebpEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let img = domain::encoder::pixels_to_dynamic_image(&pixels, &domain_info)
            .map_err(to_wit_error)?;
        let domain_config = domain::encoder::webp::WebpEncodeConfig {
            quality: config.quality.unwrap_or(75),
            lossless: config.lossless.unwrap_or(false),
        };
        domain::encoder::webp::encode(&img, &domain_info, &domain_config).map_err(to_wit_error)
    }

    fn encode(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        format: String,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::encoder::encode(&pixels, &domain_info, &format, quality).map_err(to_wit_error)
    }

    fn supported_formats() -> Vec<String> {
        domain::encoder::supported_formats()
    }
}

impl transform::Guest for Component {
    fn resize(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        width: u32,
        height: u32,
        filter: transform::ResizeFilter,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_filter = match filter {
            transform::ResizeFilter::Nearest => domain::types::ResizeFilter::Nearest,
            transform::ResizeFilter::Bilinear => domain::types::ResizeFilter::Bilinear,
            transform::ResizeFilter::Bicubic => domain::types::ResizeFilter::Bicubic,
            transform::ResizeFilter::Lanczos3 => domain::types::ResizeFilter::Lanczos3,
        };
        let result = domain::transform::resize(&pixels, &domain_info, width, height, domain_filter)
            .map_err(to_wit_error)?;
        Ok((result.pixels, to_wit_image_info(&result.info)))
    }

    fn crop(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let result = domain::transform::crop(&pixels, &domain_info, x, y, width, height)
            .map_err(to_wit_error)?;
        Ok((result.pixels, to_wit_image_info(&result.info)))
    }

    fn rotate(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        degrees: transform::Rotation,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let rotation = match degrees {
            transform::Rotation::R90 => domain::types::Rotation::R90,
            transform::Rotation::R180 => domain::types::Rotation::R180,
            transform::Rotation::R270 => domain::types::Rotation::R270,
        };
        let result =
            domain::transform::rotate(&pixels, &domain_info, rotation).map_err(to_wit_error)?;
        Ok((result.pixels, to_wit_image_info(&result.info)))
    }

    fn flip(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        direction: transform::FlipDirection,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let dir = match direction {
            transform::FlipDirection::Horizontal => domain::types::FlipDirection::Horizontal,
            transform::FlipDirection::Vertical => domain::types::FlipDirection::Vertical,
        };
        let result = domain::transform::flip(&pixels, &domain_info, dir).map_err(to_wit_error)?;
        Ok((result.pixels, to_wit_image_info(&result.info)))
    }

    fn convert_format(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        target: types::PixelFormat,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_target = to_domain_pixel_format(target);
        let result = domain::transform::convert_format(&pixels, &domain_info, domain_target)
            .map_err(to_wit_error)?;
        Ok((result.pixels, to_wit_image_info(&result.info)))
    }
}

impl filters::Guest for Component {
    fn blur(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        radius: f32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::filters::blur(&pixels, &domain_info, radius).map_err(to_wit_error)
    }

    fn sharpen(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        amount: f32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::filters::sharpen(&pixels, &domain_info, amount).map_err(to_wit_error)
    }

    fn brightness(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        amount: f32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::filters::brightness(&pixels, &domain_info, amount).map_err(to_wit_error)
    }

    fn contrast(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        amount: f32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::filters::contrast(&pixels, &domain_info, amount).map_err(to_wit_error)
    }

    fn grayscale(
        pixels: Vec<u8>,
        info: types::ImageInfo,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let result = domain::filters::grayscale(&pixels, &domain_info).map_err(to_wit_error)?;
        Ok((result.pixels, to_wit_image_info(&result.info)))
    }

    fn composite(
        fg_pixels: Vec<u8>,
        fg_info: types::ImageInfo,
        bg_pixels: Vec<u8>,
        bg_info: types::ImageInfo,
        x: i32,
        y: i32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_fg_info = to_domain_image_info(&fg_info);
        let domain_bg_info = to_domain_image_info(&bg_info);
        domain::composite::alpha_composite_over(
            &fg_pixels,
            &domain_fg_info,
            &bg_pixels,
            &domain_bg_info,
            x,
            y,
        )
        .map_err(to_wit_error)
    }
}
