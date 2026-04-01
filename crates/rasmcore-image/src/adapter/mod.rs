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

// Auto-generated stateless encoder methods (per-format encode_xxx)
include!(concat!(env!("OUT_DIR"), "/generated_encoder_adapter.rs"));

impl encoder::Guest for Component {
    // Auto-generated per-format encode methods
    generated_encoder_methods!();

    fn encode(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        format: String,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        domain::encoder::encode(&pixels, &domain_info, &format, quality).map_err(to_wit_error)
    }

    fn encode_with_icc(
        pixels: Vec<u8>,
        info: types::ImageInfo,
        format: String,
        quality: Option<u8>,
        icc_profile: Vec<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_info = to_domain_image_info(&info);
        let encoded =
            domain::encoder::encode(&pixels, &domain_info, &format, quality).map_err(to_wit_error)?;
        domain::encoder::embed_icc_profile(&encoded, &format, &icc_profile).map_err(to_wit_error)
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

// (generated_encoder_adapter.rs included earlier, before impl encoder::Guest)

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

fn to_wit_metadata_set(ms: &domain::metadata::set::MetadataSet) -> metadata::MetadataSet {
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

fn to_domain_metadata_set(ms: &metadata::MetadataSet) -> domain::metadata::set::MetadataSet {
    domain::metadata::set::MetadataSet {
        exif: ms.exif.clone(),
        xmp: ms.xmp.clone(),
        iptc: ms.iptc.clone(),
        icc_profile: ms.icc_profile.clone(),
        format_specific: ms
            .format_specific
            .iter()
            .map(|c| domain::metadata::set::MetadataChunk {
                key: c.key.clone(),
                value: c.value.clone(),
            })
            .collect(),
    }
}

// Compare interface — auto-generated from #[register_metric] annotations
// (included via generated_compare_adapter.rs above)
