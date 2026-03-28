//! JPEG marker segment reading and writing.
//!
//! ITU-T T.81 Section B. Handles all JPEG marker segments:
//! - SOI (Start of Image) / EOI (End of Image)
//! - SOF0-SOF15 (Start of Frame — baseline, progressive, etc.)
//! - SOS (Start of Scan)
//! - DHT (Define Huffman Table)
//! - DQT (Define Quantization Table)
//! - DRI (Define Restart Interval)
//! - RST0-RST7 (Restart markers)
//! - APP0-APP15 (Application segments — JFIF, EXIF, XMP, ICC)
//! - COM (Comment)
//!
//! Also handles metadata embedding via APP markers (unified with MetadataSet).

// Stub — implementation in a future track.
