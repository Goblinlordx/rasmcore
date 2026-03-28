//! Integration tests for the pipeline engine.

#[cfg(test)]
mod tests {
    use crate::domain::pipeline::graph::NodeGraph;
    use crate::domain::pipeline::nodes::filters::{BlurNode, GrayscaleNode};
    use crate::domain::pipeline::nodes::sink;
    use crate::domain::pipeline::nodes::source::SourceNode;
    use crate::domain::pipeline::nodes::transform::{CropNode, FlipNode, ResizeNode, RotateNode};
    use crate::domain::pipeline::rect::Rect;
    use crate::domain::types::*;

    fn make_test_png() -> Vec<u8> {
        let img = image::RgbImage::from_fn(64, 64, |x, y| {
            image::Rgb([(x * 4) as u8, (y * 4) as u8, 128])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn pipeline_read_write_roundtrip() {
        let png_data = make_test_png();
        let mut graph = NodeGraph::new(4 * 1024 * 1024);

        let src = graph.add_node(Box::new(SourceNode::new(png_data).unwrap()));
        let output = sink::write(&mut graph, src, "png", None).unwrap();

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

        let output = sink::write(&mut graph, resized, "png", None).unwrap();
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

        let output = sink::write(&mut graph, cropped, "png", None).unwrap();
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

        let blurred = graph.add_node(Box::new(BlurNode::new(resized, resized_info, 2.0)));

        let output = sink::write(&mut graph, blurred, "jpeg", Some(85)).unwrap();
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

        let thumb_out = sink::write(&mut graph, thumb, "png", None).unwrap();
        let large_out = sink::write(&mut graph, large, "png", None).unwrap();

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
        let gray = graph.add_node(Box::new(GrayscaleNode::new(src, src_info)));

        assert_eq!(graph.node_info(gray).unwrap().format, PixelFormat::Gray8);

        let output = sink::write(&mut graph, gray, "png", None).unwrap();
        let decoded = crate::domain::decoder::decode(&output).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Gray8);
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

        let output = sink::write(&mut graph, flipped, "png", None).unwrap();
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
}
