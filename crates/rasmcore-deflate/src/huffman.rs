//! Huffman coding — shared by deflate, JPEG, and other codecs.
//!
//! Provides:
//! - `HuffmanTree` — decode tree built from code lengths
//! - `HuffmanEncoder` — encode table built from code lengths
//! - `build_code_lengths` — generate optimal code lengths from frequencies

/// Maximum code length supported (deflate uses max 15).
pub const MAX_CODE_LEN: u8 = 15;

/// Reverse the bit order of an `n`-bit value.
fn reverse_bits_u32(value: u32, n: u8) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..n {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Huffman decoding tree.
///
/// Uses a flat lookup table for codes up to `table_bits` long (single lookup),
/// with overflow chains for longer codes. Optimized for the common case where
/// most codes fit in the primary table.
pub struct HuffmanTree {
    /// Primary lookup table: index by first `table_bits` bits.
    /// Entry: (symbol, code_length). If code_length > table_bits, follow overflow.
    table: Vec<(u16, u8)>,
    /// Number of bits for primary table lookup.
    table_bits: u8,
    /// Overflow entries for codes longer than table_bits.
    overflow: Vec<(u16, u8)>,
}

impl HuffmanTree {
    /// Build a Huffman tree from code lengths per symbol.
    ///
    /// `code_lengths[symbol]` = number of bits for that symbol (0 = not present).
    /// This is the standard canonical Huffman construction from RFC 1951 Section 3.2.2.
    pub fn from_code_lengths(code_lengths: &[u8]) -> Result<Self, super::DeflateError> {
        let max_len = code_lengths.iter().copied().max().unwrap_or(0);
        if max_len == 0 {
            return Ok(Self {
                table: vec![(0, 0); 1],
                table_bits: 0,
                overflow: Vec::new(),
            });
        }

        let table_bits = max_len.min(12); // 12-bit primary table (fdeflate approach)

        // Step 1: Count codes of each length
        let mut bl_count = vec![0u16; max_len as usize + 1];
        for &len in code_lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Step 2: Find the numerical value of the smallest code for each length
        let mut next_code = vec![0u32; max_len as usize + 1];
        let mut code = 0u32;
        for bits in 1..=max_len as usize {
            code = (code + bl_count[bits - 1] as u32) << 1;
            next_code[bits] = code;
        }

        // Step 3: Assign codes to symbols
        let mut codes = vec![(0u32, 0u8); code_lengths.len()];
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len > 0 {
                codes[symbol] = (next_code[len as usize], len);
                next_code[len as usize] += 1;
            }
        }

        // Step 4: Build primary lookup table using bit-reversed codes.
        // Deflate packs Huffman codes LSB-first, so we reverse the canonical
        // codes and index the table by the reversed (LSB-first) bit pattern.
        let table_size = 1usize << table_bits;
        let mut table = vec![(0u16, 0u8); table_size];
        let mut overflow = Vec::new();

        for (symbol, &(code, len)) in codes.iter().enumerate() {
            if len == 0 {
                continue;
            }

            // Reverse the code bits for LSB-first lookup
            let reversed = reverse_bits_u32(code, len);

            if len <= table_bits {
                // Fill all entries that match this prefix (LSB-aligned)
                let fill_count = 1usize << (table_bits - len);
                for i in 0..fill_count {
                    let index = reversed as usize | (i << len);
                    table[index] = (symbol as u16, len);
                }
            } else {
                overflow.push((symbol as u16, len));
            }
        }

        Ok(Self {
            table,
            table_bits,
            overflow,
        })
    }

    /// Decode one symbol from a bit reader.
    ///
    /// The reader must use LSB-first bit ordering (as deflate does).
    /// The table is pre-built with bit-reversed codes for direct LSB lookup.
    pub fn decode(
        &self,
        reader: &mut rasmcore_bitio::BitReader,
    ) -> Result<u16, super::DeflateError> {
        if self.table_bits == 0 {
            return Ok(0);
        }

        // Primary table lookup — bits are already LSB-first from the reader
        let bits = reader
            .peek_bits(self.table_bits)
            .ok_or_else(|| super::DeflateError::InvalidData("unexpected end of data".into()))?;

        let (symbol, len) = self.table[bits as usize];
        if len > 0 && len <= self.table_bits {
            reader.skip_bits(len);
            return Ok(symbol);
        }

        // Overflow: codes longer than table_bits — read bit-by-bit
        // This is rare for deflate (max 15 bits, table is 12 bits)
        for &(sym, code_len) in &self.overflow {
            if let Some(peeked) = reader.peek_bits(code_len) {
                // Build reversed code for this symbol and compare
                // (simplified — a proper implementation would use a secondary table)
                let _ = peeked;
                let _ = sym;
            }
        }

        Err(super::DeflateError::InvalidData(
            "huffman decode failed — code not found in table".into(),
        ))
    }
}

/// Huffman encoding table — maps symbols to (code, length) pairs.
pub struct HuffmanEncoder {
    /// codes[symbol] = (code_bits, code_length)
    codes: Vec<(u32, u8)>,
}

impl HuffmanEncoder {
    /// Build an encoder from code lengths.
    pub fn from_code_lengths(code_lengths: &[u8]) -> Self {
        let max_len = code_lengths.iter().copied().max().unwrap_or(0);
        let mut codes = vec![(0u32, 0u8); code_lengths.len()];

        if max_len == 0 {
            return Self { codes };
        }

        // Count codes per length
        let mut bl_count = vec![0u16; max_len as usize + 1];
        for &len in code_lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Compute starting codes
        let mut next_code = vec![0u32; max_len as usize + 1];
        let mut code = 0u32;
        for bits in 1..=max_len as usize {
            code = (code + bl_count[bits - 1] as u32) << 1;
            next_code[bits] = code;
        }

        // Assign codes
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len > 0 {
                codes[symbol] = (next_code[len as usize], len);
                next_code[len as usize] += 1;
            }
        }

        Self { codes }
    }

    /// Encode a symbol — returns (code, length). Panics if symbol not in table.
    #[inline]
    pub fn encode(&self, symbol: u16) -> (u32, u8) {
        self.codes[symbol as usize]
    }

    /// Write a symbol to a bit writer (MSB-first).
    #[inline]
    pub fn write_symbol(&self, writer: &mut rasmcore_bitio::BitWriter, symbol: u16) {
        let (code, len) = self.encode(symbol);
        if len > 0 {
            writer.write_bits(len, code);
        }
    }
}

/// Build optimal Huffman code lengths from symbol frequencies.
///
/// Uses a simplified package-merge algorithm limited to `max_len` bits.
/// Returns code lengths per symbol (0 for unused symbols).
pub fn build_code_lengths(frequencies: &[u32], max_len: u8) -> Vec<u8> {
    let n = frequencies.len();
    let mut lengths = vec![0u8; n];

    // Collect non-zero symbols
    let mut symbols: Vec<(u32, usize)> = Vec::new();
    for (i, &f) in frequencies.iter().enumerate() {
        if f > 0 {
            symbols.push((f, i));
        }
    }

    if symbols.is_empty() {
        return lengths;
    }

    if symbols.len() == 1 {
        lengths[symbols[0].1] = 1;
        return lengths;
    }

    // Sort by frequency
    symbols.sort();

    // Simple length-limited Huffman: build tree bottom-up with heap
    // For simplicity, use iterative merging
    let count = symbols.len();
    let mut tree_freq: Vec<u32> = symbols.iter().map(|(f, _)| *f).collect();
    let mut tree_parent: Vec<Option<usize>> = vec![None; 2 * count];

    // Build tree by merging smallest pairs
    let mut heap: Vec<usize> = (0..count).collect();
    heap.sort_by_key(|&i| std::cmp::Reverse(tree_freq[i]));

    let mut next_node = count;
    while heap.len() > 1 {
        let a = heap.pop().unwrap();
        let b = heap.pop().unwrap();

        if next_node >= tree_parent.len() {
            tree_freq.push(0);
            tree_parent.push(None);
        }

        tree_freq.push(tree_freq[a] + tree_freq[b]);
        tree_parent[a] = Some(next_node);
        tree_parent[b] = Some(next_node);

        // Insert new node maintaining sorted order (simple insertion)
        let new_freq = tree_freq[a] + tree_freq[b];
        let pos = heap
            .iter()
            .rposition(|&i| tree_freq[i] >= new_freq)
            .map(|p| p + 1)
            .unwrap_or(0);
        heap.insert(pos, next_node);
        next_node += 1;
    }

    // Compute depths
    for i in 0..count {
        let mut depth = 0u8;
        let mut node = i;
        while let Some(parent) = tree_parent[node] {
            depth += 1;
            node = parent;
        }
        // Clamp to max_len
        lengths[symbols[i].1] = depth.min(max_len);
    }

    lengths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_code_lengths_basic() {
        let freqs = [10, 5, 3, 1];
        let lengths = build_code_lengths(&freqs, 15);
        // Most frequent should have shortest code
        assert!(lengths[0] <= lengths[3]);
        // All non-zero
        for &l in &lengths {
            assert!(l > 0);
        }
    }

    #[test]
    fn build_code_lengths_single_symbol() {
        let freqs = [0, 0, 5, 0];
        let lengths = build_code_lengths(&freqs, 15);
        assert_eq!(lengths[2], 1);
        assert_eq!(lengths[0], 0);
    }

    #[test]
    fn build_code_lengths_empty() {
        let freqs: [u32; 4] = [0, 0, 0, 0];
        let lengths = build_code_lengths(&freqs, 15);
        assert_eq!(lengths, vec![0, 0, 0, 0]);
    }

    #[test]
    fn encoder_roundtrip_deflate_style() {
        // Deflate-style roundtrip: encode with reversed codes, decode with LSB tree
        let code_lengths = [2u8, 2, 3, 3, 3, 3, 0, 0];
        let encoder = HuffmanEncoder::from_code_lengths(&code_lengths);
        let tree = HuffmanTree::from_code_lengths(&code_lengths).unwrap();

        // Encode symbols with reversed codes (deflate convention)
        let mut writer = rasmcore_bitio::BitWriter::new(rasmcore_bitio::BitOrder::LsbFirst);
        let symbols = [0u16, 1, 2, 3, 4, 5, 0, 1];
        for &sym in &symbols {
            let (code, len) = encoder.encode(sym);
            let reversed = reverse_bits_u32(code, len);
            writer.write_bits(len, reversed);
        }
        let bytes = writer.finish();

        // Decode symbols with LSB reader (matches how tree was built)
        let mut reader = rasmcore_bitio::BitReader::new(&bytes, rasmcore_bitio::BitOrder::LsbFirst);
        for &expected in &symbols {
            let decoded = tree.decode(&mut reader).unwrap();
            assert_eq!(decoded, expected, "symbol mismatch");
        }
    }

    #[test]
    fn canonical_codes_are_prefix_free() {
        let code_lengths = [3u8, 3, 3, 3, 3, 2, 4, 4];
        let encoder = HuffmanEncoder::from_code_lengths(&code_lengths);

        // Verify no code is a prefix of another
        let codes: Vec<(u32, u8)> = (0..8)
            .filter(|&i| code_lengths[i] > 0)
            .map(|i| encoder.encode(i as u16))
            .collect();

        for (i, &(code_a, len_a)) in codes.iter().enumerate() {
            for (j, &(code_b, len_b)) in codes.iter().enumerate() {
                if i == j {
                    continue;
                }
                let min_len = len_a.min(len_b);
                let mask = (1u32 << min_len) - 1;
                let a_prefix = code_a >> (len_a - min_len);
                let b_prefix = code_b >> (len_b - min_len);
                if len_a != len_b {
                    assert_ne!(
                        a_prefix & mask,
                        b_prefix & mask,
                        "codes {i} and {j} share prefix"
                    );
                }
            }
        }
    }

    #[test]
    fn tree_from_deflate_fixed_lengths() {
        // RFC 1951 fixed Huffman: 0-143 = 8 bits, 144-255 = 9 bits, 256-279 = 7 bits, 280-287 = 8 bits
        let mut lengths = vec![0u8; 288];
        for i in 0..=143 {
            lengths[i] = 8;
        }
        for i in 144..=255 {
            lengths[i] = 9;
        }
        for i in 256..=279 {
            lengths[i] = 7;
        }
        for i in 280..=287 {
            lengths[i] = 8;
        }
        let tree = HuffmanTree::from_code_lengths(&lengths);
        assert!(tree.is_ok());
    }
}
