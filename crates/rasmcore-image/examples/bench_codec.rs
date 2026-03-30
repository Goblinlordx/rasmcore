//! Minimal CLI for benchmarking codec performance as a process spawn.
//!
//! Usage:
//!   bench_codec decode <input_file>
//!   bench_codec encode <input_file> <format> [quality]
//!
//! Decode: reads file, decodes to pixels, discards output.
//! Encode: reads file, decodes, re-encodes to specified format, writes to stdout.

use rasmcore_image::domain::{decoder, encoder};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: bench_codec decode <file>");
        eprintln!("       bench_codec encode <file> <format> [quality]");
        std::process::exit(1);
    }

    let mode = &args[1];
    let path = &args[2];
    let data = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {path}: {e}");
        std::process::exit(1);
    });

    match mode.as_str() {
        "decode" => {
            let _dec = decoder::decode(&data).unwrap_or_else(|e| {
                eprintln!("Decode failed: {e}");
                std::process::exit(1);
            });
        }
        "encode" => {
            let fmt = &args[3];
            let quality = args.get(4).map(|q| q.parse::<u8>().unwrap());
            let dec = decoder::decode(&data).unwrap_or_else(|e| {
                eprintln!("Decode failed: {e}");
                std::process::exit(1);
            });
            let encoded =
                encoder::encode(&dec.pixels, &dec.info, fmt, quality).unwrap_or_else(|e| {
                    eprintln!("Encode failed: {e}");
                    std::process::exit(1);
                });
            // Write to stdout
            use std::io::Write;
            std::io::stdout().write_all(&encoded).unwrap();
        }
        _ => {
            eprintln!("Unknown mode: {mode}");
            std::process::exit(1);
        }
    }
}
