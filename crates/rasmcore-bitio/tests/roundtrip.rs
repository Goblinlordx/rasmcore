//! Roundtrip tests: write bits → read bits → verify exact match.

use rasmcore_bitio::{BitOrder, BitReader, BitWriter};

#[test]
fn msb_roundtrip_mixed_widths() {
    let mut w = BitWriter::new(BitOrder::MsbFirst);
    let values: Vec<(u8, u32)> = vec![
        (1, 1),
        (3, 0b101),
        (8, 0xAB),
        (16, 0x1234),
        (5, 0b10101),
        (7, 0b1100110),
    ];

    for &(n, v) in &values {
        w.write_bits(n, v);
    }
    let bytes = w.finish();

    let mut r = BitReader::new(&bytes, BitOrder::MsbFirst);
    for &(n, expected) in &values {
        let got = r.read_bits(n).unwrap();
        assert_eq!(
            got, expected,
            "mismatch at width {n}: expected {expected:#x}, got {got:#x}"
        );
    }
}

#[test]
fn lsb_roundtrip_mixed_widths() {
    let mut w = BitWriter::new(BitOrder::LsbFirst);
    let values: Vec<(u8, u32)> = vec![
        (1, 0),
        (3, 0b110),
        (9, 0x1FF),
        (12, 0xABC),
        (4, 0b0101),
        (7, 0b0011001),
    ];

    for &(n, v) in &values {
        w.write_bits(n, v);
    }
    let bytes = w.finish();

    let mut r = BitReader::new(&bytes, BitOrder::LsbFirst);
    for &(n, expected) in &values {
        let got = r.read_bits(n).unwrap();
        assert_eq!(
            got, expected,
            "mismatch at width {n}: expected {expected:#x}, got {got:#x}"
        );
    }
}

#[test]
fn msb_roundtrip_all_byte_values() {
    let mut w = BitWriter::new(BitOrder::MsbFirst);
    for i in 0..=255u32 {
        w.write_bits(8, i);
    }
    let bytes = w.finish();
    assert_eq!(bytes.len(), 256);

    let mut r = BitReader::new(&bytes, BitOrder::MsbFirst);
    for i in 0..=255u32 {
        assert_eq!(r.read_bits(8).unwrap(), i);
    }
}

#[test]
fn lsb_roundtrip_all_byte_values() {
    let mut w = BitWriter::new(BitOrder::LsbFirst);
    for i in 0..=255u32 {
        w.write_bits(8, i);
    }
    let bytes = w.finish();
    assert_eq!(bytes.len(), 256);

    let mut r = BitReader::new(&bytes, BitOrder::LsbFirst);
    for i in 0..=255u32 {
        assert_eq!(r.read_bits(8).unwrap(), i);
    }
}

#[test]
fn msb_roundtrip_single_bits_1000() {
    let mut w = BitWriter::new(BitOrder::MsbFirst);
    let pattern: Vec<bool> = (0..1000).map(|i| i % 3 == 0 || i % 7 == 0).collect();
    for &bit in &pattern {
        w.write_bit(bit);
    }
    let bytes = w.finish();

    let mut r = BitReader::new(&bytes, BitOrder::MsbFirst);
    for (i, &expected) in pattern.iter().enumerate() {
        let got = r.read_bit().unwrap();
        assert_eq!(got, expected, "bit {i} mismatch");
    }
}

#[test]
fn lsb_roundtrip_variable_width_stress() {
    let mut w = BitWriter::new(BitOrder::LsbFirst);
    let mut entries = Vec::new();
    for width in 1..=16u8 {
        let value = ((width as u32) * 17 + 3) & ((1u32 << width) - 1);
        entries.push((width, value));
        w.write_bits(width, value);
    }
    let bytes = w.finish();

    let mut r = BitReader::new(&bytes, BitOrder::LsbFirst);
    for &(width, expected) in &entries {
        let got = r.read_bits(width).unwrap();
        assert_eq!(got, expected, "width {width}");
    }
}

#[test]
fn msb_32bit_roundtrip() {
    let mut w = BitWriter::new(BitOrder::MsbFirst);
    w.write_bits(32, 0xDEADBEEF);
    let bytes = w.finish();
    assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);

    let mut r = BitReader::new(&bytes, BitOrder::MsbFirst);
    assert_eq!(r.read_bits(32).unwrap(), 0xDEADBEEF);
}
