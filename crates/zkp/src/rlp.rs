//! Ethereum Recursive Length Prefix (RLP) codec — reference implementation.
//!
//! Per Ethereum Yellow Paper Appendix B, RLP is the serialisation format used
//! for every structural object on the execution layer: block headers, account
//! records, transaction payloads, and — crucially — the nodes of the Merkle
//! Patricia Trie ([`crate::mpt`]).
//!
//! This module is a spec-complete encoder/decoder used as the *reference* the
//! future AIR will mirror byte-for-byte. It intentionally avoids pulling in any
//! external RLP crate.
//!
//! # Encoding rules (Yellow Paper §B)
//! A string `b`:
//! - If `|b| == 1` and `b[0] ∈ 0x00..0x7f`: encode as `b` itself.
//! - Else if `|b| ≤ 55`: encode as `[0x80 + |b|, b...]`.
//! - Else: encode as `[0xB7 + |len(|b|)_be|, len(|b|)_be..., b...]`.
//!
//! A list `l` whose concatenated payload has length `L`:
//! - If `L ≤ 55`: encode as `[0xC0 + L, payload...]`.
//! - Else: encode as `[0xF7 + |L_be|, L_be..., payload...]`.
//!
//! Integers are encoded as their *minimal* big-endian byte representation
//! (leading zeros stripped; the integer `0` encodes as the empty string).

use thiserror::Error;

/// Decoded RLP item. Mirrors the encoder's two cases exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RlpItem {
    /// A byte string.
    Bytes(Vec<u8>),
    /// A list of nested items.
    List(Vec<RlpItem>),
}

/// Errors produced while decoding an RLP byte stream.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RlpError {
    /// Input buffer ended before the expected number of bytes were available.
    #[error("rlp: unexpected end of input")]
    UnexpectedEof,
    /// A length prefix was not in minimal canonical form
    /// (e.g. single byte < 0x80 wrapped in `0x81, …`, or long-form length
    /// with a leading zero byte).
    #[error("rlp: non-canonical length encoding")]
    NonCanonical,
    /// A length-of-length prefix exceeded the platform `usize`.
    #[error("rlp: length overflow")]
    LengthOverflow,
}

// -----------------------------------------------------------------------------
// Encoding
// -----------------------------------------------------------------------------

/// Encode `b` as a raw RLP byte string.
pub fn rlp_encode_bytes(b: &[u8]) -> Vec<u8> {
    if b.len() == 1 && b[0] < 0x80 {
        return vec![b[0]];
    }
    if b.len() <= 55 {
        let mut out = Vec::with_capacity(1 + b.len());
        out.push(0x80 + b.len() as u8);
        out.extend_from_slice(b);
        return out;
    }
    let len_be = minimal_be(b.len() as u64);
    let mut out = Vec::with_capacity(1 + len_be.len() + b.len());
    out.push(0xB7 + len_be.len() as u8);
    out.extend_from_slice(&len_be);
    out.extend_from_slice(b);
    out
}

/// Encode a list whose items have already been RLP-encoded.
///
/// `items` is the sequence of encoded items; this function concatenates them
/// and prepends the appropriate list-length prefix.
pub fn rlp_encode_list(items: &[Vec<u8>]) -> Vec<u8> {
    let payload_len: usize = items.iter().map(|i| i.len()).sum();
    let mut out = if payload_len <= 55 {
        let mut v = Vec::with_capacity(1 + payload_len);
        v.push(0xC0 + payload_len as u8);
        v
    } else {
        let len_be = minimal_be(payload_len as u64);
        let mut v = Vec::with_capacity(1 + len_be.len() + payload_len);
        v.push(0xF7 + len_be.len() as u8);
        v.extend_from_slice(&len_be);
        v
    };
    for item in items {
        out.extend_from_slice(item);
    }
    out
}

/// Encode an unsigned integer as its minimal big-endian byte string, then RLP.
/// Zero encodes as the empty string `0x80`.
pub fn rlp_encode_uint(n: u64) -> Vec<u8> {
    rlp_encode_bytes(&minimal_be(n))
}

/// Encode a 32-byte big-endian unsigned integer (U256) with leading zeros
/// stripped (Ethereum's convention for RLP'd 256-bit quantities).
pub fn rlp_encode_u256(be_bytes: &[u8; 32]) -> Vec<u8> {
    let start = be_bytes.iter().position(|&b| b != 0).unwrap_or(be_bytes.len());
    rlp_encode_bytes(&be_bytes[start..])
}

/// Minimal big-endian encoding of `n` (no leading zeros; zero → empty).
fn minimal_be(n: u64) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    let full = n.to_be_bytes();
    let start = full.iter().position(|&b| b != 0).unwrap();
    full[start..].to_vec()
}

// -----------------------------------------------------------------------------
// Decoding
// -----------------------------------------------------------------------------

/// Decode a single RLP item from the head of `input`.
/// Returns the item plus the total number of bytes consumed.
pub fn rlp_decode(input: &[u8]) -> Result<(RlpItem, usize), RlpError> {
    if input.is_empty() {
        return Err(RlpError::UnexpectedEof);
    }
    let tag = input[0];
    match tag {
        // Single byte in [0x00, 0x7f]: the byte itself is the string.
        0x00..=0x7f => Ok((RlpItem::Bytes(vec![tag]), 1)),

        // Short string: 0..55 bytes.
        0x80..=0xB7 => {
            let len = (tag - 0x80) as usize;
            if 1 + len > input.len() {
                return Err(RlpError::UnexpectedEof);
            }
            let body = &input[1..1 + len];
            // Canonicity: a 1-byte string whose sole byte is < 0x80 must use
            // the single-byte form.
            if len == 1 && body[0] < 0x80 {
                return Err(RlpError::NonCanonical);
            }
            Ok((RlpItem::Bytes(body.to_vec()), 1 + len))
        }

        // Long string: length-of-length in [1, 8].
        0xB8..=0xBF => {
            let len_of_len = (tag - 0xB7) as usize;
            if 1 + len_of_len > input.len() {
                return Err(RlpError::UnexpectedEof);
            }
            let len = decode_be_length(&input[1..1 + len_of_len])?;
            if len <= 55 {
                return Err(RlpError::NonCanonical);
            }
            let header = 1 + len_of_len;
            if header + len > input.len() {
                return Err(RlpError::UnexpectedEof);
            }
            let body = &input[header..header + len];
            Ok((RlpItem::Bytes(body.to_vec()), header + len))
        }

        // Short list: payload length 0..55.
        0xC0..=0xF7 => {
            let len = (tag - 0xC0) as usize;
            if 1 + len > input.len() {
                return Err(RlpError::UnexpectedEof);
            }
            let items = decode_list_payload(&input[1..1 + len])?;
            Ok((RlpItem::List(items), 1 + len))
        }

        // Long list: length-of-length in [1, 8].
        0xF8..=0xFF => {
            let len_of_len = (tag - 0xF7) as usize;
            if 1 + len_of_len > input.len() {
                return Err(RlpError::UnexpectedEof);
            }
            let len = decode_be_length(&input[1..1 + len_of_len])?;
            if len <= 55 {
                return Err(RlpError::NonCanonical);
            }
            let header = 1 + len_of_len;
            if header + len > input.len() {
                return Err(RlpError::UnexpectedEof);
            }
            let items = decode_list_payload(&input[header..header + len])?;
            Ok((RlpItem::List(items), header + len))
        }
    }
}

fn decode_list_payload(mut payload: &[u8]) -> Result<Vec<RlpItem>, RlpError> {
    let mut items = Vec::new();
    while !payload.is_empty() {
        let (item, consumed) = rlp_decode(payload)?;
        items.push(item);
        payload = &payload[consumed..];
    }
    Ok(items)
}

fn decode_be_length(bytes: &[u8]) -> Result<usize, RlpError> {
    if bytes.is_empty() {
        return Err(RlpError::NonCanonical);
    }
    if bytes[0] == 0 {
        return Err(RlpError::NonCanonical);
    }
    if bytes.len() > std::mem::size_of::<usize>() {
        return Err(RlpError::LengthOverflow);
    }
    let mut len: usize = 0;
    for &b in bytes {
        len = len.checked_shl(8).ok_or(RlpError::LengthOverflow)?;
        len |= b as usize;
    }
    Ok(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Single-byte / short-string canonical vectors -----------------------

    #[test]
    fn test_encode_dog() {
        assert_eq!(rlp_encode_bytes(b"dog"), vec![0x83, b'd', b'o', b'g']);
    }

    #[test]
    fn test_encode_empty_string() {
        assert_eq!(rlp_encode_bytes(b""), vec![0x80]);
    }

    #[test]
    fn test_encode_zero_byte() {
        assert_eq!(rlp_encode_bytes(&[0x00]), vec![0x00]);
    }

    #[test]
    fn test_encode_single_small_byte() {
        assert_eq!(rlp_encode_bytes(&[0x0f]), vec![0x0f]);
    }

    #[test]
    fn test_encode_two_bytes_high() {
        assert_eq!(rlp_encode_bytes(&[0x04, 0x00]), vec![0x82, 0x04, 0x00]);
    }

    #[test]
    fn test_encode_single_byte_0x80() {
        // 0x80 itself is NOT a single-byte case (only <0x80 is); must use 0x81 prefix.
        assert_eq!(rlp_encode_bytes(&[0x80]), vec![0x81, 0x80]);
    }

    // --- Integers -----------------------------------------------------------

    #[test]
    fn test_encode_uint_zero() {
        assert_eq!(rlp_encode_uint(0), vec![0x80]);
    }

    #[test]
    fn test_encode_uint_one() {
        assert_eq!(rlp_encode_uint(1), vec![0x01]);
    }

    #[test]
    fn test_encode_uint_1024() {
        // 1024 = 0x0400 (big-endian).
        assert_eq!(rlp_encode_uint(1024), vec![0x82, 0x04, 0x00]);
    }

    #[test]
    fn test_encode_u256_small() {
        let mut be = [0u8; 32];
        be[31] = 0x2a; // 42
        assert_eq!(rlp_encode_u256(&be), vec![0x2a]);
    }

    #[test]
    fn test_encode_u256_zero() {
        let be = [0u8; 32];
        assert_eq!(rlp_encode_u256(&be), vec![0x80]);
    }

    // --- Lists --------------------------------------------------------------

    #[test]
    fn test_encode_empty_list() {
        assert_eq!(rlp_encode_list(&[]), vec![0xc0]);
    }

    #[test]
    fn test_encode_cat_dog_list() {
        let items = vec![
            rlp_encode_bytes(b"cat"),
            rlp_encode_bytes(b"dog"),
        ];
        let enc = rlp_encode_list(&items);
        assert_eq!(
            enc,
            vec![0xc8, 0x83, b'c', b'a', b't', 0x83, b'd', b'o', b'g']
        );
    }

    // --- Long string -------------------------------------------------------

    #[test]
    fn test_encode_56_byte_string() {
        let s = vec![0xABu8; 56];
        let enc = rlp_encode_bytes(&s);
        assert_eq!(enc[0], 0xB8);
        assert_eq!(enc[1], 0x38);
        assert_eq!(enc.len(), 2 + 56);
        assert_eq!(&enc[2..], &s[..]);
    }

    #[test]
    fn test_encode_long_string_1024() {
        let s = vec![0x7eu8; 1024];
        let enc = rlp_encode_bytes(&s);
        // 1024 = 0x04 0x00, two length bytes → prefix 0xB7 + 2 = 0xB9.
        assert_eq!(enc[0], 0xB9);
        assert_eq!(enc[1], 0x04);
        assert_eq!(enc[2], 0x00);
        assert_eq!(enc.len(), 3 + 1024);
    }

    // --- Decoding -----------------------------------------------------------

    #[test]
    fn test_decode_dog() {
        let enc = rlp_encode_bytes(b"dog");
        let (item, n) = rlp_decode(&enc).unwrap();
        assert_eq!(n, enc.len());
        assert_eq!(item, RlpItem::Bytes(b"dog".to_vec()));
    }

    #[test]
    fn test_decode_empty_string() {
        let (item, n) = rlp_decode(&[0x80]).unwrap();
        assert_eq!(n, 1);
        assert_eq!(item, RlpItem::Bytes(Vec::new()));
    }

    #[test]
    fn test_decode_single_byte() {
        let (item, n) = rlp_decode(&[0x0f]).unwrap();
        assert_eq!(n, 1);
        assert_eq!(item, RlpItem::Bytes(vec![0x0f]));
    }

    #[test]
    fn test_decode_empty_list() {
        let (item, n) = rlp_decode(&[0xc0]).unwrap();
        assert_eq!(n, 1);
        assert_eq!(item, RlpItem::List(Vec::new()));
    }

    #[test]
    fn test_decode_cat_dog_list() {
        let enc = vec![0xc8, 0x83, b'c', b'a', b't', 0x83, b'd', b'o', b'g'];
        let (item, n) = rlp_decode(&enc).unwrap();
        assert_eq!(n, enc.len());
        assert_eq!(
            item,
            RlpItem::List(vec![
                RlpItem::Bytes(b"cat".to_vec()),
                RlpItem::Bytes(b"dog".to_vec()),
            ])
        );
    }

    #[test]
    fn test_decode_long_string() {
        let s = vec![0xABu8; 56];
        let enc = rlp_encode_bytes(&s);
        let (item, n) = rlp_decode(&enc).unwrap();
        assert_eq!(n, enc.len());
        assert_eq!(item, RlpItem::Bytes(s));
    }

    #[test]
    fn test_decode_nested_list() {
        // [ "dog", [ "cat" ], [] ]
        let enc = rlp_encode_list(&[
            rlp_encode_bytes(b"dog"),
            rlp_encode_list(&[rlp_encode_bytes(b"cat")]),
            rlp_encode_list(&[]),
        ]);
        let (item, n) = rlp_decode(&enc).unwrap();
        assert_eq!(n, enc.len());
        assert_eq!(
            item,
            RlpItem::List(vec![
                RlpItem::Bytes(b"dog".to_vec()),
                RlpItem::List(vec![RlpItem::Bytes(b"cat".to_vec())]),
                RlpItem::List(Vec::new()),
            ])
        );
    }

    #[test]
    fn test_decode_truncated_string() {
        // 0x83 says 3-byte string follows, but only 2 are present.
        let buf = vec![0x83, b'd', b'o'];
        assert_eq!(rlp_decode(&buf).unwrap_err(), RlpError::UnexpectedEof);
    }

    #[test]
    fn test_decode_non_canonical_single_byte() {
        // 0x81 0x00: should have been 0x00 directly.
        let buf = vec![0x81, 0x00];
        assert_eq!(rlp_decode(&buf).unwrap_err(), RlpError::NonCanonical);
    }

    #[test]
    fn test_decode_non_canonical_long_string_too_short() {
        // 0xB8 0x05 "hello": length-of-length form with length 5 ≤ 55 should
        // have used the short form 0x85 instead.
        let buf = vec![0xB8, 0x05, b'h', b'e', b'l', b'l', b'o'];
        assert_eq!(rlp_decode(&buf).unwrap_err(), RlpError::NonCanonical);
    }

    #[test]
    fn test_decode_non_canonical_leading_zero_length() {
        // 0xB9 0x00 0x56 ...  — 56 bytes but with a leading zero in the length
        // prefix. Not canonical.
        let mut buf = vec![0xB9, 0x00, 0x38];
        buf.extend(std::iter::repeat(0xAB).take(56));
        assert_eq!(rlp_decode(&buf).unwrap_err(), RlpError::NonCanonical);
    }

    // --- Round-trip --------------------------------------------------------

    #[test]
    fn test_round_trip_bytes() {
        let cases: &[&[u8]] = &[
            b"",
            b"\x00",
            b"\x0f",
            b"dog",
            &[0x04, 0x00],
            &[0xABu8; 56],
            &[0x7eu8; 1024],
        ];
        for &c in cases {
            let enc = rlp_encode_bytes(c);
            let (item, n) = rlp_decode(&enc).unwrap();
            assert_eq!(n, enc.len());
            assert_eq!(item, RlpItem::Bytes(c.to_vec()));
        }
    }

    #[test]
    fn test_round_trip_nested_lists() {
        // A moderately deep, mixed structure.
        let inner = rlp_encode_list(&[
            rlp_encode_bytes(b"a"),
            rlp_encode_uint(0),
            rlp_encode_uint(0xDEAD_BEEF),
        ]);
        let outer = rlp_encode_list(&[
            inner.clone(),
            rlp_encode_bytes(&[0xFFu8; 100]),
            rlp_encode_list(&[]),
        ]);
        let (item, n) = rlp_decode(&outer).unwrap();
        assert_eq!(n, outer.len());
        match item {
            RlpItem::List(items) => assert_eq!(items.len(), 3),
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn test_minimal_be() {
        assert_eq!(minimal_be(0), Vec::<u8>::new());
        assert_eq!(minimal_be(1), vec![0x01]);
        assert_eq!(minimal_be(0xff), vec![0xff]);
        assert_eq!(minimal_be(0x100), vec![0x01, 0x00]);
        assert_eq!(
            minimal_be(0x0123_4567_89ab_cdef),
            vec![0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]
        );
    }
}
