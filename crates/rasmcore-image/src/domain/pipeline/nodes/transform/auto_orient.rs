use crate::domain::error::ImageError;
use crate::domain::metadata::ExifOrientation;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform;
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

/// Auto-orient node — applies EXIF orientation transform.
pub struct AutoOrientNode {
    upstream: u32,
    orientation: ExifOrientation,
    source_info: ImageInfo,
}

impl AutoOrientNode {
    pub fn new(upstream: u32, source_info: ImageInfo, orientation: ExifOrientation) -> Self {
        Self {
            upstream,
            orientation,
            source_info,
        }
    }
}

#[rasmcore_macros::register_transform(name = "auto_orient")]
impl ImageNode for AutoOrientNode {
    fn info(&self) -> ImageInfo {
        let (w, h) = match self.orientation {
            ExifOrientation::Normal
            | ExifOrientation::FlipHorizontal
            | ExifOrientation::Rotate180
            | ExifOrientation::FlipVertical => (self.source_info.width, self.source_info.height),
            ExifOrientation::Transpose
            | ExifOrientation::Rotate90
            | ExifOrientation::Transverse
            | ExifOrientation::Rotate270 => (self.source_info.height, self.source_info.width),
        };
        ImageInfo {
            width: w,
            height: h,
            ..self.source_info.clone()
        }
    }

    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let full_src = Rect::new(0, 0, self.source_info.width, self.source_info.height);
        let src_pixels = upstream_fn(self.upstream, full_src)?;
        let result = transform::auto_orient(&src_pixels, &self.source_info, self.orientation)?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }

    fn derive_metadata(
        &self,
        upstream: &rasmcore_pipeline::Metadata,
    ) -> Option<rasmcore_pipeline::Metadata> {
        let info = self.info();
        let had_orientation = upstream.exif_orientation().is_some();
        let dims_changed =
            info.width != self.source_info.width || info.height != self.source_info.height;

        if !had_orientation && !dims_changed {
            return None;
        }

        let mut meta = upstream.clone();
        if had_orientation {
            meta.set("exif.Orientation", rasmcore_pipeline::MetadataValue::Int(1));
        }
        if dims_changed {
            meta.set("width", rasmcore_pipeline::MetadataValue::Int(info.width as i64));
            meta.set("height", rasmcore_pipeline::MetadataValue::Int(info.height as i64));
        }
        Some(meta)
    }
}
