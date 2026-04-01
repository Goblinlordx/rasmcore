//! Integration tests for the pipeline engine.

#[cfg(test)]
mod tests {
    use crate::domain::filters::{BlurParams, CharcoalParams};
    use crate::domain::pipeline::graph::NodeGraph;
    use crate::domain::pipeline::nodes::composite::CompositeNode;
    use crate::domain::pipeline::nodes::filters::{
        BlurNode, CharcoalMapperNode, GrayscaleMapperNode,
    };
    use crate::domain::pipeline::nodes::sink;
    use crate::domain::pipeline::nodes::source::SourceNode;
    use crate::domain::pipeline::nodes::transform::{CropNode, FlipNode, ResizeNode, RotateNode};
    use crate::domain::types::*;
    use rasmcore_pipeline::Rect;

    fn make_test_png() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64 * 64)
            .flat_map(|i| {
                let x = i % 64;
                let y = i / 64;
                [(x * 4) as u8, (y * 4) as u8, 128u8]
            })
            .collect();
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        crate::domain::encoder::png::encode(
            &pixels,
            &info,
            &crate::domain::encoder::png::PngEncodeConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn pipeline_read_write_roundtrip() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let output = sink::write(&mut graph, src, "png", None, None).unwrap();

        // Output should be valid PNG
        assert_eq!(&output[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn pipeline_read_resize_write() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let resized = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info,
            32,
            16,
            ResizeFilter::Lanczos3,
        )));

        assert_eq!(graph.node_info(resized).unwrap().width, 32);
        assert_eq!(graph.node_info(resized).unwrap().height, 16);

        let output = sink::write(&mut graph, resized, "png", None, None).unwrap();
        // Decode the output and verify dimensions
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.width, 32);
        assert_eq!(decoded.info.height, 16);
    }

    #[test]
    fn pipeline_read_crop_write() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let cropped = graph.add_node(Box::new(CropNode::new(src, src_info, 10, 10, 20, 20)));

        let output = sink::write(&mut graph, cropped, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.width, 20);
        assert_eq!(decoded.info.height, 20);
    }

    #[test]
    fn pipeline_multi_step_chain() {
        // read → resize → blur → write
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        let resized = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info,
            32,
            32,
            ResizeFilter::Lanczos3,
        )));
        let resized_info = graph.node_info(resized).unwrap();

        let blurred = graph.add_node(Box::new(BlurNode::new(
            resized,
            resized_info,
            BlurParams { radius: 2.0 },
        )));

        let output = sink::write(&mut graph, blurred, "jpeg", Some(85), None).unwrap();
        // Should produce valid JPEG
        assert_eq!(&output[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn pipeline_dag_same_source_two_outputs() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        // Two different resizes from the same source
        let thumb = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info.clone(),
            16,
            16,
            ResizeFilter::Nearest,
        )));
        let large = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info,
            48,
            48,
            ResizeFilter::Lanczos3,
        )));

        let thumb_out = sink::write(&mut graph, thumb, "png", None, None).unwrap();
        let large_out = sink::write(&mut graph, large, "png", None, None).unwrap();

        let thumb_dec = crate::domain::decoder::decode(&thumb_out).unwrap();
        let large_dec = crate::domain::decoder::decode(&large_out).unwrap();
        assert_eq!(thumb_dec.info.width, 16);
        assert_eq!(large_dec.info.width, 48);
    }

    #[test]
    fn pipeline_grayscale_chain() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let gray = graph.add_node(Box::new(GrayscaleMapperNode::new(src, src_info)));

        // After first compute, info() should return Gray8
        // (Before compute it returns source_info as conservative estimate)
        let output = sink::write(&mut graph, gray, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Gray8);
    }

    #[test]
    fn pipeline_charcoal_blur_chain() {
        // Charcoal outputs Gray8 — verify downstream blur works correctly on Gray8
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();

        // Charcoal mapper: RGB8 → Gray8
        let charcoal_node = graph.add_node(Box::new(CharcoalMapperNode::new(
            src,
            src_info,
            CharcoalParams {
                radius: 1.0,
                sigma: 0.5,
            },
        )));

        // Query charcoal's output info — should be Gray8 thanks to output_format
        let charcoal_out_info = graph.node_info(charcoal_node).unwrap();
        assert_eq!(charcoal_out_info.format, PixelFormat::Gray8);

        // Blur on Gray8 output — this was broken before the mapper fix
        let blurred = graph.add_node(Box::new(BlurNode::new(
            charcoal_node,
            charcoal_out_info,
            BlurParams { radius: 1.0 },
        )));

        let output = sink::write(&mut graph, blurred, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        // Final output should be Gray8 (charcoal converts, blur preserves)
        assert_eq!(decoded.info.format, PixelFormat::Gray8);
        // Verify dimensions preserved
        assert_eq!(decoded.info.width, 64);
        assert_eq!(decoded.info.height, 64);
    }

    #[test]
    fn pipeline_rotate_flip_chain() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let src_info = graph.node_info(src).unwrap();
        let rotated = graph.add_node(Box::new(RotateNode::new(src, src_info, Rotation::R90)));
        let rotated_info = graph.node_info(rotated).unwrap();
        let flipped = graph.add_node(Box::new(FlipNode::new(
            rotated,
            rotated_info,
            FlipDirection::Horizontal,
        )));

        let output = sink::write(&mut graph, flipped, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        // 64x64 rotated 90 = 64x64 (square), flipped = 64x64
        assert_eq!(decoded.info.width, 64);
        assert_eq!(decoded.info.height, 64);
    }

    #[test]
    fn pipeline_cache_reuse() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));

        // Request full region
        let full = Rect::new(0, 0, 64, 64);
        let p1 = graph.request_region(src, full).unwrap();

        // Request again — should be cached
        let p2 = graph.request_region(src, full).unwrap();
        assert_eq!(p1, p2);

        // Request sub-region — should extract from cache
        let sub = graph
            .request_region(src, Rect::new(10, 10, 20, 20))
            .unwrap();
        let src_info = graph.node_info(src).unwrap();
        let bpp = match src_info.format {
            PixelFormat::Rgb8 => 3usize,
            PixelFormat::Rgba8 => 4,
            _ => 4,
        };
        assert_eq!(sub.len(), 20 * 20 * bpp);
    }

    fn make_test_rgba_png(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let pixels: Vec<u8> = (0..(w * h)).flat_map(|_| [r, g, b, a]).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        crate::domain::encoder::png::encode(
            &pixels,
            &info,
            &crate::domain::encoder::png::PngEncodeConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn pipeline_composite_opaque_watermark() {
        // Composite a small red overlay onto a blue background
        let bg_png = make_test_rgba_png(64, 64, 0, 0, 255, 255);
        let fg_png = make_test_rgba_png(16, 16, 255, 0, 0, 255);

        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let bg = graph.add_node(Box::new(SourceNode::new(bg_png).unwrap()));
        let fg = graph.add_node(Box::new(SourceNode::new(fg_png).unwrap()));

        let bg_info = graph.node_info(bg).unwrap();
        let fg_info = graph.node_info(fg).unwrap();

        let comp = graph.add_node(Box::new(CompositeNode::new(
            fg, bg, fg_info, bg_info, 10, 10, None,
        )));

        // Output should have bg dimensions
        assert_eq!(graph.node_info(comp).unwrap().width, 64);
        assert_eq!(graph.node_info(comp).unwrap().height, 64);

        let output = sink::write(&mut graph, comp, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.width, 64);
        assert_eq!(decoded.info.height, 64);
    }

    #[test]
    fn pipeline_composite_pixel_correctness() {
        // Verify exact pixel values after compositing
        let bg_png = make_test_rgba_png(4, 4, 0, 0, 255, 255); // blue
        let fg_png = make_test_rgba_png(2, 2, 255, 0, 0, 128); // 50% red

        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let bg = graph.add_node(Box::new(SourceNode::new(bg_png).unwrap()));
        let fg = graph.add_node(Box::new(SourceNode::new(fg_png).unwrap()));

        let bg_info = graph.node_info(bg).unwrap();
        let fg_info = graph.node_info(fg).unwrap();

        let comp = graph.add_node(Box::new(CompositeNode::new(
            fg, bg, fg_info, bg_info, 1, 1, None,
        )));

        // Request full region
        let pixels = graph.request_region(comp, Rect::new(0, 0, 4, 4)).unwrap();

        // Pixel (0,0): should be pure blue (no fg overlay)
        assert_eq!(&pixels[0..4], [0, 0, 255, 255]);

        // Pixel (1,1): should be blended (50% red over blue)
        let px = &pixels[(1 * 4 + 1) * 4..(1 * 4 + 1) * 4 + 4];
        assert_eq!(px[0], 128); // red channel
        assert_eq!(px[1], 0); // green
        assert_eq!(px[2], 127); // blue channel
        assert_eq!(px[3], 255); // alpha

        // Pixel (3,3): should be pure blue (outside fg)
        let px = &pixels[(3 * 4 + 3) * 4..(3 * 4 + 3) * 4 + 4];
        assert_eq!(px, [0, 0, 255, 255]);
    }

    #[test]
    fn pipeline_composite_chain() {
        // read bg → read fg → resize fg → composite → write
        let bg_png = make_test_rgba_png(64, 64, 0, 0, 255, 255);
        let fg_png = make_test_rgba_png(32, 32, 255, 0, 0, 255);

        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let bg = graph.add_node(Box::new(SourceNode::new(bg_png).unwrap()));
        let fg = graph.add_node(Box::new(SourceNode::new(fg_png).unwrap()));

        let fg_info = graph.node_info(fg).unwrap();
        let resized_fg = graph.add_node(Box::new(ResizeNode::new(
            fg,
            fg_info,
            16,
            16,
            ResizeFilter::Lanczos3,
        )));

        let bg_info = graph.node_info(bg).unwrap();
        let resized_fg_info = graph.node_info(resized_fg).unwrap();

        let comp = graph.add_node(Box::new(CompositeNode::new(
            resized_fg,
            bg,
            resized_fg_info,
            bg_info,
            24,
            24,
            None,
        )));

        let output = sink::write(&mut graph, comp, "png", None, None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.width, 64);
        assert_eq!(decoded.info.height, 64);
    }

    #[test]
    fn pre_execution_validation_catches_empty_graph() {
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let result = sink::write(&mut graph, 0, "png", None, None);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("empty graph"),
            "expected empty graph error, got: {err}"
        );
    }

    #[test]
    fn pre_execution_validation_passes_for_valid_pipeline() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        // Should succeed — graph is valid
        let output = sink::write(&mut graph, src, "png", None, None).unwrap();
        assert_eq!(&output[..4], &[0x89, b'P', b'N', b'G']);
    }
}
