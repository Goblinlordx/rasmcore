//! WASM adapter layer — thin WIT binding glue.
//! Only compiled for wasm32 targets.

mod pipeline_adapter;

use crate::bindings;
use crate::bindings::exports::rasmcore::image::{
    compare, decoder, encoder, filters, metadata, pipeline, transform,
};
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
        types::PixelFormat::Rgb16 => domain::types::PixelFormat::Rgb16,
        types::PixelFormat::Rgba16 => domain::types::PixelFormat::Rgba16,
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
        domain::types::PixelFormat::Rgb16 => types::PixelFormat::Rgb16,
        domain::types::PixelFormat::Rgba16 => types::PixelFormat::Rgba16,
        domain::types::PixelFormat::Yuv420p => types::PixelFormat::Yuv420p,
        domain::types::PixelFormat::Yuv422p => types::PixelFormat::Yuv422p,
        domain::types::PixelFormat::Yuv444p => types::PixelFormat::Yuv444p,
        domain::types::PixelFormat::Nv12 => types::PixelFormat::Nv12,
        domain::types::PixelFormat::Cmyk8 => types::PixelFormat::Rgb8, // CMYK decoded to RGB before WIT
        domain::types::PixelFormat::Cmyka8 => types::PixelFormat::Rgba8, // CMYKA decoded to RGBA before WIT
    }
}

fn to_wit_color_space(c: domain::types::ColorSpace) -> types::ColorSpace {
    match c {
        domain::types::ColorSpace::Srgb => types::ColorSpace::Srgb,
        domain::types::ColorSpace::LinearSrgb => types::ColorSpace::LinearSrgb,
        domain::types::ColorSpace::DisplayP3 => types::ColorSpace::DisplayP3,
        domain::types::ColorSpace::Bt709 => types::ColorSpace::Bt709,
        domain::types::ColorSpace::Bt2020 => types::ColorSpace::Bt2020,
        domain::types::ColorSpace::ProPhotoRgb => types::ColorSpace::ProphotoRgb,
        domain::types::ColorSpace::AdobeRgb => types::ColorSpace::AdobeRgb,
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
            types::ColorSpace::ProphotoRgb => domain::types::ColorSpace::ProPhotoRgb,
            types::ColorSpace::AdobeRgb => domain::types::ColorSpace::AdobeRgb,
        },
    }
}

fn to_domain_tiff_compression(
    c: Option<encoder::TiffCompression>,
) -> domain::encoder::tiff::TiffCompression {
    match c {
        None => domain::encoder::tiff::TiffCompression::Lzw,
        Some(encoder::TiffCompression::None) => domain::encoder::tiff::TiffCompression::None,
        Some(encoder::TiffCompression::Lzw) => domain::encoder::tiff::TiffCompression::Lzw,
        Some(encoder::TiffCompression::Deflate) => domain::encoder::tiff::TiffCompression::Deflate,
        Some(encoder::TiffCompression::Packbits) => {
            domain::encoder::tiff::TiffCompression::PackBits
        }
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

fn to_wit_disposal_method(d: domain::types::DisposalMethod) -> decoder::DisposalMethod {
    match d {
        domain::types::DisposalMethod::None => decoder::DisposalMethod::None,
        domain::types::DisposalMethod::Background => decoder::DisposalMethod::Background,
        domain::types::DisposalMethod::Previous => decoder::DisposalMethod::Previous,
    }
}

fn to_wit_frame_info(fi: &domain::types::FrameInfo) -> decoder::FrameInfo {
    decoder::FrameInfo {
        index: fi.index,
        delay_ms: fi.delay_ms,
        disposal: to_wit_disposal_method(fi.disposal),
        width: fi.width,
        height: fi.height,
        x_offset: fi.x_offset,
        y_offset: fi.y_offset,
    }
}

fn to_wit_decoded_frame(
    img: domain::types::DecodedImage,
    fi: domain::types::FrameInfo,
) -> decoder::DecodedFrame {
    decoder::DecodedFrame {
        image: decoder::DecodedImage {
            pixels: img.pixels,
            info: to_wit_image_info(&img.info),
            icc_profile: img.icc_profile,
        },
        info: to_wit_frame_info(&fi),
    }
}

pub fn to_domain_frame_selection(s: pipeline::FrameSelection) -> domain::types::FrameSelection {
    match s {
        pipeline::FrameSelection::Single(i) => domain::types::FrameSelection::Single(i),
        pipeline::FrameSelection::Pick(v) => domain::types::FrameSelection::Pick(v),
        pipeline::FrameSelection::Range((start, end)) => {
            domain::types::FrameSelection::Range(start, end)
        }
        pipeline::FrameSelection::All => domain::types::FrameSelection::All,
    }
}

fn from_wit_disposal_method(d: decoder::DisposalMethod) -> domain::types::DisposalMethod {
    match d {
        decoder::DisposalMethod::None => domain::types::DisposalMethod::None,
        decoder::DisposalMethod::Background => domain::types::DisposalMethod::Background,
        decoder::DisposalMethod::Previous => domain::types::DisposalMethod::Previous,
    }
}

fn wit_frames_to_sequence(
    frames: Vec<encoder::EncodeFrame>,
    canvas_width: u32,
    canvas_height: u32,
) -> domain::types::FrameSequence {
    let mut seq = domain::types::FrameSequence::new(canvas_width, canvas_height);
    for f in frames {
        let image = domain::types::DecodedImage {
            pixels: f.pixels,
            info: to_domain_image_info(&f.info),
            icc_profile: None,
        };
        let fi = domain::types::FrameInfo {
            index: f.frame_info.index,
            delay_ms: f.frame_info.delay_ms,
            disposal: from_wit_disposal_method(f.frame_info.disposal),
            width: f.frame_info.width,
            height: f.frame_info.height,
            x_offset: f.frame_info.x_offset,
            y_offset: f.frame_info.y_offset,
        };
        seq.push(image, fi);
    }
    seq
}

struct Component;

bindings::export!(Component with_types_in bindings);

impl pipeline::Guest for Component {
    type ImagePipeline = pipeline_adapter::PipelineResource;
    type LayerCache = pipeline_adapter::LayerCacheResource;

    fn detect_format(header: Vec<u8>) -> Option<String> {
        domain::decoder::detect_format(&header)
    }

    fn supported_read_formats() -> Vec<String> {
        domain::decoder::supported_formats()
    }

    fn supported_write_formats() -> Vec<String> {
        domain::encoder::supported_formats()
    }

    fn get_filter_manifest() -> String {
        include_str!(concat!(env!("OUT_DIR"), "/param-manifest.json")).to_string()
    }

    fn get_manifest_hash() -> String {
        include_str!(concat!(env!("OUT_DIR"), "/param-manifest.hash")).to_string()
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
            icc_profile: result.icc_profile,
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
            icc_profile: result.icc_profile,
        })
    }

    fn supported_formats() -> Vec<String> {
        domain::decoder::supported_formats()
    }

    fn frame_count(data: Vec<u8>) -> Result<u32, RasmcoreError> {
        domain::decoder::frame_count(&data).map_err(to_wit_error)
    }

    fn decode_frame(data: Vec<u8>, index: u32) -> Result<decoder::DecodedFrame, RasmcoreError> {
        let (img, fi) = domain::decoder::decode_frame(&data, index).map_err(to_wit_error)?;
        Ok(to_wit_decoded_frame(img, fi))
    }

    fn decode_all_frames(data: Vec<u8>) -> Result<Vec<decoder::DecodedFrame>, RasmcoreError> {
        let frames = domain::decoder::decode_all_frames(&data).map_err(to_wit_error)?;
        Ok(frames
            .into_iter()
            .map(|(img, fi)| to_wit_decoded_frame(img, fi))
            .collect())
    }
}

impl encoder::Guest for Component {
    fn encode_jpeg(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::JpegEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_config = domain::encoder::jpeg::JpegEncodeConfig {
            quality: config.quality.unwrap_or(85),
            progressive: config.progressive.unwrap_or(false),
            turbo: false,
        };
        domain::encoder::jpeg::encode_pixels(&pixels, &domain_info, &domain_config)
            .map_err(to_wit_error)
    }

    fn encode_jpeg_with_icc(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::JpegEncodeConfig,
        icc_profile: Vec<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_config = domain::encoder::jpeg::JpegEncodeConfig {
            quality: config.quality.unwrap_or(85),
            progressive: config.progressive.unwrap_or(false),
            turbo: false,
        };
        let encoded = domain::encoder::jpeg::encode_pixels(&pixels, &domain_info, &domain_config)
            .map_err(to_wit_error)?;
        domain::encoder::jpeg::embed_icc_profile(&encoded, &icc_profile).map_err(to_wit_error)
    }

    fn encode_png(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::PngEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_config = domain::encoder::png::PngEncodeConfig {
            compression_level: config.compression_level.unwrap_or(6),
            filter_type: to_domain_png_filter(config.filter_type),
        };
        domain::encoder::png::encode(&pixels, &domain_info, &domain_config).map_err(to_wit_error)
    }

    fn encode_png_with_icc(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::PngEncodeConfig,
        icc_profile: Vec<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_config = domain::encoder::png::PngEncodeConfig {
            compression_level: config.compression_level.unwrap_or(6),
            filter_type: to_domain_png_filter(config.filter_type),
        };
        let encoded = domain::encoder::png::encode(&pixels, &domain_info, &domain_config)
            .map_err(to_wit_error)?;
        domain::encoder::png::embed_icc_profile(&encoded, &icc_profile).map_err(to_wit_error)
    }

    fn encode_webp(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::WebpEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_config = domain::encoder::webp::WebpEncodeConfig {
            quality: config.quality.unwrap_or(75),
            lossless: config.lossless.unwrap_or(false),
        };
        domain::encoder::webp::encode_pixels(&pixels, &domain_info, &domain_config)
            .map_err(to_wit_error)
    }

    fn encode_avif(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::AvifEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_config = domain::encoder::avif::AvifEncodeConfig {
            quality: config.quality.unwrap_or(75),
            speed: config.speed.unwrap_or(6),
        };
        domain::encoder::avif::encode(&pixels, &domain_info, &domain_config).map_err(to_wit_error)
    }

    fn encode_tiff(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::TiffEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_config = domain::encoder::tiff::TiffEncodeConfig {
            compression: to_domain_tiff_compression(config.compression),
        };
        domain::encoder::tiff::encode(&pixels, &domain_info, &domain_config).map_err(to_wit_error)
    }

    fn encode_bmp(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        _config: encoder::BmpEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::encoder::bmp::encode_pixels(
            &pixels,
            &domain_info,
            &domain::encoder::bmp::BmpEncodeConfig,
        )
        .map_err(to_wit_error)
    }

    fn encode_ico(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        _config: encoder::IcoEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::encoder::ico::encode_pixels(
            &pixels,
            &domain_info,
            &domain::encoder::ico::IcoEncodeConfig,
        )
        .map_err(to_wit_error)
    }

    fn encode_qoi(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        _config: encoder::QoiEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::encoder::encode(&pixels, &domain_info, "qoi", None).map_err(to_wit_error)
    }

    fn encode_gif(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        config: encoder::GifEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_config = domain::encoder::gif::GifEncodeConfig {
            repeat: config.repeat.unwrap_or(0),
        };
        domain::encoder::gif::encode_pixels(&pixels, &domain_info, &domain_config)
            .map_err(to_wit_error)
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

    fn get_format_info(format: String) -> Option<types::FormatInfo> {
        domain::encoder::format_info(&format).map(|fi| types::FormatInfo {
            name: fi.name,
            mime_type: fi.mime_type,
            extensions: fi.extensions,
        })
    }

    fn all_format_info() -> Vec<types::FormatInfo> {
        domain::encoder::all_format_info()
            .into_iter()
            .map(|fi| types::FormatInfo {
                name: fi.name,
                mime_type: fi.mime_type,
                extensions: fi.extensions,
            })
            .collect()
    }

    fn encode_gif_sequence(
        frames: Vec<encoder::EncodeFrame>,
        canvas_width: u32,
        canvas_height: u32,
        config: encoder::GifEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let seq = wit_frames_to_sequence(frames, canvas_width, canvas_height);
        let domain_config = domain::encoder::gif::GifEncodeConfig {
            repeat: config.repeat.unwrap_or(0),
        };
        domain::encoder::gif::encode_sequence(&seq, &domain_config).map_err(to_wit_error)
    }

    fn encode_tiff_pages(
        frames: Vec<encoder::EncodeFrame>,
        config: encoder::TiffEncodeConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let canvas = frames
            .first()
            .map(|f| (f.info.width, f.info.height))
            .unwrap_or((0, 0));
        let seq = wit_frames_to_sequence(frames, canvas.0, canvas.1);
        let domain_config = domain::encoder::tiff::TiffEncodeConfig {
            compression: to_domain_tiff_compression(config.compression),
        };
        domain::encoder::tiff::encode_pages(&seq, &domain_config).map_err(to_wit_error)
    }

    fn encode_apng_sequence(
        frames: Vec<encoder::EncodeFrame>,
        canvas_width: u32,
        canvas_height: u32,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let seq = wit_frames_to_sequence(frames, canvas_width, canvas_height);
        let config = domain::encoder::png::PngEncodeConfig::default();
        domain::encoder::png::encode_sequence(&seq, &config).map_err(to_wit_error)
    }

    fn encode_sequence(
        frames: Vec<encoder::EncodeFrame>,
        canvas_width: u32,
        canvas_height: u32,
        format: String,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let seq = wit_frames_to_sequence(frames, canvas_width, canvas_height);
        domain::encoder::encode_sequence(&seq, &format, quality).map_err(to_wit_error)
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

    fn auto_orient(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        orientation: transform::ExifOrientation,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let domain_orient = to_domain_exif_orientation(orientation);
        let result = domain::transform::auto_orient(&pixels, &domain_info, domain_orient)
            .map_err(to_wit_error)?;
        Ok((result.pixels, to_wit_image_info(&result.info)))
    }

    fn auto_orient_from_exif(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        encoded_data: Vec<u8>,
    ) -> Result<(Vec<u8>, types::ImageInfo), RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let result = domain::transform::auto_orient_from_exif(&pixels, &domain_info, &encoded_data)
            .map_err(to_wit_error)?;
        Ok((result.pixels, to_wit_image_info(&result.info)))
    }
}

// Auto-generated filter adapter (all registered filters)
include!(concat!(env!("OUT_DIR"), "/generated_filter_adapter.rs"));

// Auto-generated compare adapter (all registered metrics)
include!(concat!(env!("OUT_DIR"), "/generated_compare_adapter.rs"));

fn to_domain_exif_orientation(o: metadata::ExifOrientation) -> domain::metadata::ExifOrientation {
    match o {
        metadata::ExifOrientation::Normal => domain::metadata::ExifOrientation::Normal,
        metadata::ExifOrientation::FlipHorizontal => {
            domain::metadata::ExifOrientation::FlipHorizontal
        }
        metadata::ExifOrientation::Rotate180 => domain::metadata::ExifOrientation::Rotate180,
        metadata::ExifOrientation::FlipVertical => domain::metadata::ExifOrientation::FlipVertical,
        metadata::ExifOrientation::Transpose => domain::metadata::ExifOrientation::Transpose,
        metadata::ExifOrientation::Rotate90 => domain::metadata::ExifOrientation::Rotate90,
        metadata::ExifOrientation::Transverse => domain::metadata::ExifOrientation::Transverse,
        metadata::ExifOrientation::Rotate270 => domain::metadata::ExifOrientation::Rotate270,
    }
}

fn to_wit_exif_orientation(o: domain::metadata::ExifOrientation) -> metadata::ExifOrientation {
    match o {
        domain::metadata::ExifOrientation::Normal => metadata::ExifOrientation::Normal,
        domain::metadata::ExifOrientation::FlipHorizontal => {
            metadata::ExifOrientation::FlipHorizontal
        }
        domain::metadata::ExifOrientation::Rotate180 => metadata::ExifOrientation::Rotate180,
        domain::metadata::ExifOrientation::FlipVertical => metadata::ExifOrientation::FlipVertical,
        domain::metadata::ExifOrientation::Transpose => metadata::ExifOrientation::Transpose,
        domain::metadata::ExifOrientation::Rotate90 => metadata::ExifOrientation::Rotate90,
        domain::metadata::ExifOrientation::Transverse => metadata::ExifOrientation::Transverse,
        domain::metadata::ExifOrientation::Rotate270 => metadata::ExifOrientation::Rotate270,
    }
}

impl metadata::Guest for Component {
    fn read_exif(data: Vec<u8>) -> Result<metadata::ExifMetadata, RasmcoreError> {
        let meta = domain::metadata::read_exif(&data).map_err(to_wit_error)?;
        Ok(metadata::ExifMetadata {
            orientation: meta.orientation.map(to_wit_exif_orientation),
            width: meta.width,
            height: meta.height,
            camera_make: meta.camera_make,
            camera_model: meta.camera_model,
            date_time: meta.date_time,
            software: meta.software,
        })
    }

    fn has_exif(data: Vec<u8>) -> bool {
        domain::metadata::has_exif(&data)
    }

    fn read_metadata(data: Vec<u8>) -> Result<metadata::MetadataSet, RasmcoreError> {
        let ms = domain::metadata::read_metadata(&data).map_err(to_wit_error)?;
        Ok(to_wit_metadata_set(&ms))
    }
}

fn to_wit_metadata_set(ms: &domain::metadata_set::MetadataSet) -> metadata::MetadataSet {
    metadata::MetadataSet {
        exif: ms.exif.clone(),
        xmp: ms.xmp.clone(),
        iptc: ms.iptc.clone(),
        icc_profile: ms.icc_profile.clone(),
        format_specific: ms
            .format_specific
            .iter()
            .map(|c| metadata::MetadataChunk {
                key: c.key.clone(),
                value: c.value.clone(),
            })
            .collect(),
    }
}

fn to_domain_metadata_set(ms: &metadata::MetadataSet) -> domain::metadata_set::MetadataSet {
    domain::metadata_set::MetadataSet {
        exif: ms.exif.clone(),
        xmp: ms.xmp.clone(),
        iptc: ms.iptc.clone(),
        icc_profile: ms.icc_profile.clone(),
        format_specific: ms
            .format_specific
            .iter()
            .map(|c| domain::metadata_set::MetadataChunk {
                key: c.key.clone(),
                value: c.value.clone(),
            })
            .collect(),
    }
}

// Compare interface — auto-generated from #[register_metric] annotations
// (included via generated_compare_adapter.rs above)
