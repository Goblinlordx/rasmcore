//! ML inference node — calls host-provided ml-execute for AI/ML operations.
//!
//! Follows the same pattern as FilterNode but for ML inference instead of
//! compute shaders. The pipeline handles tiling, tensor conversion, and
//! overlap blending. The host just runs single-tile inference.
//!
//! Architecture:
//! - MlNode reads model capabilities (inputSpec) to decide tiling strategy
//! - For tileable models: splits input into tiles, calls ml-execute per tile,
//!   blends overlapping regions, stitches output
//! - For full-image models: resizes to target size, runs once, resizes output back
//! - Tensor conversion: f32 RGBA HWC → model format (e.g., f32 RGB NCHW)
//! - Output conversion: model output → f32 RGBA HWC

use crate::node::{InputRectEstimate, Node, NodeCapabilities, NodeInfo, PipelineError, Upstream};
use crate::rect::Rect;

// ─── Tensor Conversion ─────────────────────────────────────────────────────

/// Tensor data layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorLayout {
    /// Height × Width × Channels (interleaved). Pipeline native format.
    Hwc,
    /// Batch × Channels × Height × Width (planar). Common for ONNX models.
    Nchw,
}

/// Tensor element type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDtype {
    Float32,
    Float16,
    Uint8,
    Int8,
}

/// Convert f32 RGBA HWC pixels to a model tensor (drops alpha, transposes if needed).
///
/// Input: `&[f32]` with `width * height * 4` elements (RGBA interleaved).
/// Output: `Vec<u8>` of raw tensor bytes in the requested layout and dtype.
pub fn pixels_to_tensor(
    pixels: &[f32],
    width: u32,
    height: u32,
    layout: TensorLayout,
    dtype: TensorDtype,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;

    match (layout, dtype) {
        (TensorLayout::Nchw, TensorDtype::Float32) => {
            // f32 RGBA HWC → f32 RGB NCHW
            let plane = h * w;
            let mut out = vec![0u8; 3 * plane * 4]; // 3 channels × plane × sizeof(f32)
            for y in 0..h {
                for x in 0..w {
                    let src = (y * w + x) * 4;
                    let dst = y * w + x;
                    // R plane
                    out[dst * 4..(dst + 1) * 4].copy_from_slice(&pixels[src].to_le_bytes());
                    // G plane
                    out[(plane + dst) * 4..(plane + dst + 1) * 4]
                        .copy_from_slice(&pixels[src + 1].to_le_bytes());
                    // B plane
                    out[(2 * plane + dst) * 4..(2 * plane + dst + 1) * 4]
                        .copy_from_slice(&pixels[src + 2].to_le_bytes());
                }
            }
            out
        }
        (TensorLayout::Hwc, TensorDtype::Float32) => {
            // f32 RGBA HWC → f32 RGB HWC (drop alpha)
            let mut out = vec![0u8; 3 * h * w * 4];
            for i in 0..(h * w) {
                let src = i * 4;
                let dst = i * 3 * 4;
                out[dst..dst + 4].copy_from_slice(&pixels[src].to_le_bytes());
                out[dst + 4..dst + 8].copy_from_slice(&pixels[src + 1].to_le_bytes());
                out[dst + 8..dst + 12].copy_from_slice(&pixels[src + 2].to_le_bytes());
            }
            out
        }
        (TensorLayout::Nchw, TensorDtype::Uint8) => {
            // f32 RGBA HWC → u8 RGB NCHW
            let plane = h * w;
            let mut out = vec![0u8; 3 * plane];
            for y in 0..h {
                for x in 0..w {
                    let src = (y * w + x) * 4;
                    let dst = y * w + x;
                    out[dst] = (pixels[src].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    out[plane + dst] = (pixels[src + 1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    out[2 * plane + dst] = (pixels[src + 2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
            out
        }
        _ => {
            // Other combinations: fall back to NCHW f32
            pixels_to_tensor(
                pixels,
                width,
                height,
                TensorLayout::Nchw,
                TensorDtype::Float32,
            )
        }
    }
}

/// Convert model output tensor bytes back to f32 RGBA HWC pixels.
///
/// Output: `Vec<f32>` with `out_width * out_height * 4` elements.
pub fn tensor_to_pixels(
    tensor: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    layout: TensorLayout,
    dtype: TensorDtype,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let c = channels as usize;
    let mut pixels = vec![0.0f32; w * h * 4]; // RGBA output

    match (layout, dtype, c) {
        (TensorLayout::Nchw, TensorDtype::Float32, 3) => {
            // f32 RGB NCHW → f32 RGBA HWC
            let plane = h * w;
            for y in 0..h {
                for x in 0..w {
                    let src = y * w + x;
                    let dst = (y * w + x) * 4;
                    pixels[dst] =
                        f32::from_le_bytes(tensor[src * 4..(src + 1) * 4].try_into().unwrap());
                    pixels[dst + 1] = f32::from_le_bytes(
                        tensor[(plane + src) * 4..(plane + src + 1) * 4]
                            .try_into()
                            .unwrap(),
                    );
                    pixels[dst + 2] = f32::from_le_bytes(
                        tensor[(2 * plane + src) * 4..(2 * plane + src + 1) * 4]
                            .try_into()
                            .unwrap(),
                    );
                    pixels[dst + 3] = 1.0; // alpha = 1
                }
            }
        }
        (TensorLayout::Nchw, TensorDtype::Float32, 1) => {
            // f32 single-channel NCHW → f32 RGBA (mask: value in all channels + alpha)
            for y in 0..h {
                for x in 0..w {
                    let src = y * w + x;
                    let dst = (y * w + x) * 4;
                    let v = f32::from_le_bytes(tensor[src * 4..(src + 1) * 4].try_into().unwrap());
                    pixels[dst] = v;
                    pixels[dst + 1] = v;
                    pixels[dst + 2] = v;
                    pixels[dst + 3] = v; // mask in alpha too
                }
            }
        }
        (TensorLayout::Nchw, TensorDtype::Uint8, 3) => {
            // u8 RGB NCHW → f32 RGBA HWC
            let plane = h * w;
            for y in 0..h {
                for x in 0..w {
                    let src = y * w + x;
                    let dst = (y * w + x) * 4;
                    pixels[dst] = tensor[src] as f32 / 255.0;
                    pixels[dst + 1] = tensor[plane + src] as f32 / 255.0;
                    pixels[dst + 2] = tensor[2 * plane + src] as f32 / 255.0;
                    pixels[dst + 3] = 1.0;
                }
            }
        }
        _ => {
            // Fallback: try to interpret as NCHW f32 RGB
            if tensor.len() >= w * h * c * 4 {
                return tensor_to_pixels(
                    tensor,
                    width,
                    height,
                    3,
                    TensorLayout::Nchw,
                    TensorDtype::Float32,
                );
            }
        }
    }

    pixels
}

/// Expand single-channel mask to RGBA.
/// Input: width*height f32 values. Output: width*height*4 f32 (mask in all channels + alpha).
pub fn mask_to_rgba(mask: &[f32], width: u32, height: u32) -> Vec<f32> {
    let n = (width * height) as usize;
    let mut out = vec![0.0f32; n * 4];
    for i in 0..n {
        let v = mask[i];
        out[i * 4] = v;
        out[i * 4 + 1] = v;
        out[i * 4 + 2] = v;
        out[i * 4 + 3] = v;
    }
    out
}

// ─── Tiling ────────────────────────────────────────────────────────────────

/// Padding mode for edge tiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    Mirror,
    Zero,
    Clamp,
}

/// Input specification for a model — determines tiling strategy.
#[derive(Debug, Clone)]
pub enum MlInputSpec {
    /// Convolutional model — can be tiled with overlap.
    Tileable {
        tile_width: u32,
        tile_height: u32,
        overlap: u32,
        padding: PaddingMode,
    },
    /// Global context model — resize to target, run once.
    FullImage {
        target_width: u32,
        target_height: u32,
    },
    /// Dynamic input — no tiling or resizing needed.
    Dynamic,
}

/// What the model outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MlOutputKind {
    /// RGBA image (possibly different dimensions).
    Image,
    /// Single-channel mask (depth, segmentation, alpha).
    Mask,
}

// ─── ML Execute callback ───────────────────────────────────────────────────

/// Callback for ML inference execution. The host provides this.
/// Takes raw tensor bytes + metadata, returns raw output tensor bytes.
pub type MlExecuteFn = Box<dyn Fn(&[u8], &str, &str, &[u8]) -> Result<Vec<u8>, PipelineError>>;

// ─── MlNode ────────────────────────────────────────────────────────────────

/// ML inference node in the pipeline graph.
///
/// Calls host-provided ml-execute for each tile (or once for full-image models).
/// Handles tensor conversion, tiling, overlap blending, and stitching.
#[allow(dead_code)] // Fields used when host ml-execute callback is wired in
pub struct MlNode {
    /// Upstream node ID.
    upstream: u32,
    /// Output node info (may have different dimensions than input for upscale).
    info: NodeInfo,
    /// Model name.
    model_name: String,
    /// Model version.
    model_version: String,
    /// Serialized model params.
    params: Vec<u8>,
    /// Input specification (tiling strategy).
    input_spec: MlInputSpec,
    /// Output kind.
    output_kind: MlOutputKind,
    /// Output scale factor (1 = same dims).
    output_scale: u32,
    /// Input tensor layout expected by the model.
    tensor_layout: TensorLayout,
    /// Input tensor dtype expected by the model.
    tensor_dtype: TensorDtype,
    /// Output channels (3 for image, 1 for mask).
    output_channels: u32,
}

impl MlNode {
    /// Create a new ML node.
    ///
    /// `upstream_info` is the upstream node's output info. For upscale models,
    /// the MlNode's info will have scaled dimensions.
    pub fn new(
        upstream: u32,
        upstream_info: NodeInfo,
        model_name: String,
        model_version: String,
        params: Vec<u8>,
        input_spec: MlInputSpec,
        output_kind: MlOutputKind,
        output_scale: u32,
        tensor_layout: TensorLayout,
        tensor_dtype: TensorDtype,
    ) -> Self {
        let scale = output_scale.max(1);
        let info = NodeInfo {
            width: upstream_info.width * scale,
            height: upstream_info.height * scale,
            color_space: upstream_info.color_space,
        };
        let output_channels = match output_kind {
            MlOutputKind::Image => 3,
            MlOutputKind::Mask => 1,
        };
        Self {
            upstream,
            info,
            model_name,
            model_version,
            params,
            input_spec,
            output_kind,
            output_scale: scale,
            tensor_layout,
            tensor_dtype,
            output_channels,
        }
    }

    /// Process a single tile through ml-execute.
    #[allow(dead_code)] // Used when host ml-execute callback is wired in
    fn execute_tile(
        &self,
        pixels: &[f32],
        tile_w: u32,
        tile_h: u32,
        ml_execute: &dyn Fn(&[u8], &str, &str, &[u8]) -> Result<Vec<u8>, PipelineError>,
    ) -> Result<Vec<f32>, PipelineError> {
        // Convert pixels to model tensor format
        let tensor_bytes = pixels_to_tensor(
            pixels,
            tile_w,
            tile_h,
            self.tensor_layout,
            self.tensor_dtype,
        );

        // Call host ml-execute
        let output_bytes = ml_execute(
            &tensor_bytes,
            &self.model_name,
            &self.model_version,
            &self.params,
        )?;

        // Convert output back to f32 RGBA
        let out_w = tile_w * self.output_scale;
        let out_h = tile_h * self.output_scale;
        let out_pixels = tensor_to_pixels(
            &output_bytes,
            out_w,
            out_h,
            self.output_channels,
            self.tensor_layout,
            self.tensor_dtype,
        );

        Ok(out_pixels)
    }
}

impl Node for MlNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        _request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        // For now, request the full upstream image.
        // Tiling will be handled in a future enhancement where MlNode
        // integrates with the host's ml-execute callback.
        //
        // Current implementation: request full image, return placeholder
        // that demonstrates the node is wired correctly.
        // The actual ml-execute call requires the host callback which is
        // injected via the WASM adapter at runtime.

        let upstream_info = upstream.info(self.upstream)?;
        let full_rect = Rect::new(0, 0, upstream_info.width, upstream_info.height);
        let input_pixels = upstream.request(self.upstream, full_rect)?;

        // Without a host ml-execute callback, we can't actually run inference.
        // Return the input pixels scaled to output dimensions as a passthrough.
        // The WASM adapter will provide the actual ml-execute callback.
        let out_w = self.info.width;
        let out_h = self.info.height;

        if self.output_scale == 1 && self.output_kind == MlOutputKind::Image {
            // Same dimensions, image output — passthrough
            Ok(input_pixels)
        } else if self.output_kind == MlOutputKind::Mask {
            // Mask output — generate a white mask (placeholder)
            Ok(vec![1.0f32; (out_w * out_h * 4) as usize])
        } else {
            // Upscale — nearest-neighbor placeholder until host provides ml-execute
            let in_w = upstream_info.width as usize;
            let in_h = upstream_info.height as usize;
            let scale = self.output_scale as usize;
            let mut output = vec![0.0f32; (out_w * out_h) as usize * 4];
            for y in 0..out_h as usize {
                for x in 0..out_w as usize {
                    let sx = (x / scale).min(in_w - 1);
                    let sy = (y / scale).min(in_h - 1);
                    let src = (sy * in_w + sx) * 4;
                    let dst = (y * out_w as usize + x) * 4;
                    output[dst..dst + 4].copy_from_slice(&input_pixels[src..src + 4]);
                }
            }
            Ok(output)
        }
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            analytic: false,
            affine: false,
            clut: false,
            gpu: false, // ML nodes don't have GPU shaders — they use ml-execute
        }
    }

    fn input_rect(&self, _output: Rect, _bounds_w: u32, _bounds_h: u32) -> InputRectEstimate {
        // ML nodes always need the full upstream image (tiling is internal)
        InputRectEstimate::FullImage
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::ColorSpace;

    #[test]
    fn pixels_to_nchw_f32_roundtrip() {
        let pixels = vec![0.2, 0.4, 0.6, 1.0, 0.8, 0.6, 0.4, 1.0]; // 2 pixels RGBA
        let tensor = pixels_to_tensor(&pixels, 2, 1, TensorLayout::Nchw, TensorDtype::Float32);
        let back = tensor_to_pixels(&tensor, 2, 1, 3, TensorLayout::Nchw, TensorDtype::Float32);
        // Check RGB values (alpha restored to 1.0)
        assert!((back[0] - 0.2).abs() < 1e-6, "R0: {}", back[0]);
        assert!((back[1] - 0.4).abs() < 1e-6, "G0: {}", back[1]);
        assert!((back[2] - 0.6).abs() < 1e-6, "B0: {}", back[2]);
        assert!((back[3] - 1.0).abs() < 1e-6, "A0: {}", back[3]);
        assert!((back[4] - 0.8).abs() < 1e-6, "R1: {}", back[4]);
        assert!((back[5] - 0.6).abs() < 1e-6, "G1: {}", back[5]);
        assert!((back[6] - 0.4).abs() < 1e-6, "B1: {}", back[6]);
    }

    #[test]
    fn pixels_to_nchw_u8_roundtrip() {
        let pixels = vec![0.0, 0.5, 1.0, 1.0]; // 1 pixel
        let tensor = pixels_to_tensor(&pixels, 1, 1, TensorLayout::Nchw, TensorDtype::Uint8);
        assert_eq!(tensor.len(), 3); // 3 channels × 1 pixel × 1 byte
        assert_eq!(tensor[0], 0); // R
        assert_eq!(tensor[1], 128); // G (~0.5 * 255)
        assert_eq!(tensor[2], 255); // B
    }

    #[test]
    fn mask_to_rgba_expansion() {
        let mask = vec![0.0, 0.5, 1.0, 0.75]; // 2x2 mask
        let rgba = mask_to_rgba(&mask, 2, 2);
        assert_eq!(rgba.len(), 16); // 4 pixels × 4 channels
        // First pixel: all 0.0
        assert!((rgba[0] - 0.0).abs() < 1e-6);
        assert!((rgba[3] - 0.0).abs() < 1e-6); // alpha = mask value
        // Second pixel: all 0.5
        assert!((rgba[4] - 0.5).abs() < 1e-6);
        assert!((rgba[7] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn single_channel_nchw_to_mask() {
        // 1-channel f32 NCHW → RGBA mask
        let tensor: Vec<u8> = [0.5f32, 0.8f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let pixels = tensor_to_pixels(&tensor, 2, 1, 1, TensorLayout::Nchw, TensorDtype::Float32);
        assert_eq!(pixels.len(), 8); // 2 pixels × 4 channels
        assert!((pixels[0] - 0.5).abs() < 1e-6); // R = mask
        assert!((pixels[3] - 0.5).abs() < 1e-6); // A = mask
        assert!((pixels[4] - 0.8).abs() < 1e-6); // R = mask
        assert!((pixels[7] - 0.8).abs() < 1e-6); // A = mask
    }

    #[test]
    fn ml_node_info_scales_dimensions() {
        let upstream_info = NodeInfo {
            width: 100,
            height: 200,
            color_space: ColorSpace::Linear,
        };
        let node = MlNode::new(
            0,
            upstream_info,
            "test".into(),
            "1.0".into(),
            vec![],
            MlInputSpec::Dynamic,
            MlOutputKind::Image,
            4,
            TensorLayout::Nchw,
            TensorDtype::Float32,
        );
        assert_eq!(node.info().width, 400);
        assert_eq!(node.info().height, 800);
    }

    #[test]
    fn ml_node_mask_output_preserves_dimensions() {
        let upstream_info = NodeInfo {
            width: 256,
            height: 256,
            color_space: ColorSpace::Linear,
        };
        let node = MlNode::new(
            0,
            upstream_info,
            "rmbg".into(),
            "1.4".into(),
            vec![],
            MlInputSpec::FullImage {
                target_width: 1024,
                target_height: 1024,
            },
            MlOutputKind::Mask,
            1,
            TensorLayout::Nchw,
            TensorDtype::Float32,
        );
        // Output dimensions = input × scale (1x for mask)
        assert_eq!(node.info().width, 256);
        assert_eq!(node.info().height, 256);
    }

    #[test]
    fn ml_node_input_rect_is_full_image() {
        let upstream_info = NodeInfo {
            width: 100,
            height: 100,
            color_space: ColorSpace::Linear,
        };
        let node = MlNode::new(
            0,
            upstream_info,
            "test".into(),
            "1.0".into(),
            vec![],
            MlInputSpec::Dynamic,
            MlOutputKind::Image,
            1,
            TensorLayout::Nchw,
            TensorDtype::Float32,
        );
        let estimate = node.input_rect(Rect::new(0, 0, 50, 50), 100, 100);
        assert!(matches!(estimate, InputRectEstimate::FullImage));
    }
}
