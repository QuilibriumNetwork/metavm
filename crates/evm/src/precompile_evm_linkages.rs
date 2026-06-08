//! EVM-side cross-AIR LogUp descriptor builders binding the zkp-side
//! precompile AIRs to `precompile_air` / `precompile_io_air` (which
//! live in this crate).
//!
//! # Motivation
//!
//! Several zkp-side precompile AIRs
//! ([`metavm_zkp::kzg_point_eval_air`](metavm_zkp::kzg_point_eval_air),
//! [`metavm_zkp::ripemd160_precompile_air`](metavm_zkp::ripemd160_precompile_air),
//! [`metavm_zkp::ecrecover_chain_air`](metavm_zkp::ecrecover_chain_air),
//! [`metavm_zkp::bn254_precompile_air`](metavm_zkp::bn254_precompile_air))
//! ship stub linkage descriptors with B-side column indices set to
//! [`usize::MAX`] because the zkp crate cannot import from
//! `metavm-evm` (the dependency edge points the other way).
//!
//! Those stubs document the **intent** of the binding. This module
//! supplies the **real** descriptor builders on the EVM side that
//! plug in the actual `precompile_air` / `precompile_io_air` column
//! offsets — `metavm-evm` already depends on `metavm-zkp`, so it can
//! reach both sides' constants.
//!
//! ## What's bound
//!
//! For each precompile AIR we expose two builders:
//!
//!   1. `make_<P>_to_precompile_dispatch_descriptor` — binds the
//!      precompile AIR's per-row `(input_length, output_length,
//!      gas_cost, …)` dispatch tuple to the matching
//!      [`crate::precompile_air`] row, gated by the appropriate
//!      `sel_*` selector on both sides.
//!   2. `make_<P>_to_precompile_io_descriptor` — binds the precompile
//!      AIR's input/output byte ranges to the corresponding
//!      [`crate::precompile_io_air`] byte windows, gated by `is_real`.
//!
//! ## Caveats
//!
//! - [`crate::precompile_io_air::MAX_INPUT_LENGTH`] is currently `64`,
//!   so precompiles with wider inputs (BN254: `128`, ECRECOVER: `128`,
//!   KZG: `192`) cannot bind their full input window through the
//!   single-row IO scaffold. The builders below clamp at `min(A_LEN,
//!   IO_MAX_LEN)`; full coverage awaits the multi-row IO chunking AIR.
//! - The zkp-side stubs are intentionally **kept in place** to
//!   document the shape; they MUST NOT be passed to `joint_prove`.
//!   Always prefer the descriptors built here when wiring real
//!   joint-prove orchestration.
//! - No `joint_prove` integration tests are run here; this module is
//!   a thin descriptor-builder layer + structural validation.

use crate::precompile_air as pa;
use crate::precompile_io_air as pio;
use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

// ─── KZG point evaluation (callee 0x0a) ───────────────────────────────

use metavm_zkp::kzg_point_eval_air as kzg;

/// Bind the KZG-point-evaluation AIR dispatch row
/// `(input_length = 192, output_length = 64, gas_cost)` to the
/// corresponding [`crate::precompile_air`] row for callee `0x0a`.
///
/// The zkp-side KZG AIR does not commit dedicated length/gas columns
/// (lengths are fixed by the column layout), so we substitute the
/// AIR's `COL_IS_REAL` for each A-side slot — every real KZG row
/// implies the same constants. The B side commits the actual
/// `precompile_air` columns, gated by `sel_kzg_point`.
///
/// **Note**: This shape-only descriptor is sufficient to gate the
/// dispatch on selectors; full quantitative binding (length + gas
/// constants) requires either the zkp AIR to add literal columns, or
/// the orchestrator to inject them as fixed columns. Today the
/// selector gating alone provides the dispatch lookup.
pub fn make_kzg_eval_to_precompile_dispatch_descriptor(
    kzg_eval_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    // A side: 1-col tuple — IS_REAL acts as the "1" matching the
    // precompile-air sel_kzg_point selector on B.
    let a_columns: Vec<usize> = vec![kzg::COL_IS_REAL];
    // B side: precompile_air::COL_SEL_KZG_POINT — exactly `1` on
    // dispatched rows for callee 0x0a.
    let b_columns: Vec<usize> = vec![pa::COL_SEL_KZG_POINT];
    CrossAirLogUpDescriptor {
        label: "kzg_eval_to_precompile_dispatch_v1".into(),
        a_layer_index: kzg_eval_layer_index,
        a_columns,
        a_selector_column: Some(kzg::COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: Some(pa::COL_IS_REAL),
    }
}

/// Bind the KZG-point-evaluation AIR's `(input[0..192], output[0..64])`
/// byte arrays to the [`crate::precompile_io_air`] byte windows.
///
/// **Length clamp**: `precompile_io_air::MAX_INPUT_LENGTH = 64`, so
/// only the first 64 input bytes are bound here. Full coverage of all
/// 192 KZG input bytes requires the upcoming multi-row IO chunking
/// AIR. Output is fully bound up to
/// `min(KZG_OUTPUT, IO_SHA256_OUTPUT) = 32` bytes; the remaining 32
/// output bytes are clamped.
pub fn make_kzg_eval_to_precompile_io_descriptor(
    kzg_eval_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let input_clamp = kzg::INPUT_LEN.min(pio::MAX_INPUT_LENGTH);
    let output_clamp = kzg::OUTPUT_LEN.min(pio::SHA256_OUTPUT_LENGTH);

    let mut a_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    let mut b_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    for k in 0..input_clamp {
        a_columns.push(kzg::COL_INPUT_OFFSET + k);
        b_columns.push(pio::COL_INPUT_BYTES_OFFSET + k);
    }
    for k in 0..output_clamp {
        a_columns.push(kzg::COL_OUTPUT_OFFSET + k);
        // KZG output reuses the SHA256 output channel as the
        // fixed-width byte sink in the IO scaffold.
        b_columns.push(pio::COL_SHA256_OUTPUT_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "kzg_eval_to_precompile_io_v1_clamped".into(),
        a_layer_index: kzg_eval_layer_index,
        a_columns,
        a_selector_column: Some(kzg::COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: Some(pio::COL_IS_REAL),
    }
}

// ─── RIPEMD-160 (callee 0x03) ─────────────────────────────────────────

use metavm_zkp::ripemd160_precompile_air as ripemd;

/// Bind the RIPEMD-160 AIR's `(is_real, input_length, gas_cost)`
/// dispatch tuple to the [`crate::precompile_air`] row for callee
/// `0x03`. Output length is fixed (32) on both sides — the zkp AIR
/// does not commit it as a column, so the binding is shape-only via
/// the selectors.
pub fn make_ripemd_to_precompile_dispatch_descriptor(
    ripemd_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    // A side: (is_real as sel_ripemd-anchor, input_length, gas_cost)
    let a_columns: Vec<usize> = vec![
        ripemd::COL_IS_REAL,
        ripemd::COL_INPUT_LENGTH,
        ripemd::COL_GAS_COST,
    ];
    let b_columns: Vec<usize> = vec![
        pa::COL_SEL_RIPEMD,
        pa::COL_INPUT_LENGTH,
        pa::COL_GAS_COST,
    ];
    CrossAirLogUpDescriptor {
        label: "ripemd_to_precompile_dispatch_v1".into(),
        a_layer_index: ripemd_layer_index,
        a_columns,
        a_selector_column: Some(ripemd::COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: Some(pa::COL_IS_REAL),
    }
}

/// Bind the RIPEMD-160 AIR's input bytes window
/// `input_bytes[0..MAX_INPUT_LENGTH]` to the
/// [`crate::precompile_io_air`] input-byte window. Output (32 bytes)
/// is clamped to the IO AIR's SHA256 output channel; a dedicated
/// RIPEMD output channel awaits IO-AIR extension.
pub fn make_ripemd_to_precompile_io_descriptor(
    ripemd_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let input_clamp = ripemd::MAX_INPUT_LENGTH.min(pio::MAX_INPUT_LENGTH);
    let output_clamp = ripemd::OUTPUT_LENGTH.min(pio::SHA256_OUTPUT_LENGTH);

    let mut a_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    let mut b_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    for k in 0..input_clamp {
        a_columns.push(ripemd::COL_INPUT_BYTES_OFFSET + k);
        b_columns.push(pio::COL_INPUT_BYTES_OFFSET + k);
    }
    for k in 0..output_clamp {
        a_columns.push(ripemd::COL_OUTPUT_OFFSET + k);
        // RIPEMD output borrows the SHA-256 output channel until the
        // IO AIR adds a dedicated RIPEMD output sink.
        b_columns.push(pio::COL_SHA256_OUTPUT_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "ripemd_to_precompile_io_v1_clamped".into(),
        a_layer_index: ripemd_layer_index,
        a_columns,
        a_selector_column: Some(ripemd::COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: Some(pio::COL_IS_REAL),
    }
}

// ─── ECRECOVER (callee 0x01) ──────────────────────────────────────────

use metavm_zkp::ecrecover_chain_air as ecr;

/// Bind the ECRECOVER chain AIR's `is_real` row to the
/// [`crate::precompile_air`] `sel_ecrecover` selector.
///
/// The zkp-side AIR already exposes a parameterized builder
/// ([`ecr::make_ecrecover_to_precompile_dispatch_descriptor`]) that
/// accepts the B-side columns; this is a thin wrapper that plugs in
/// the real `precompile_air` column constants so callers don't need to
/// know them.
pub fn make_ecrecover_to_precompile_dispatch_descriptor(
    ecrecover_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    ecr::make_ecrecover_to_precompile_dispatch_descriptor(
        ecrecover_layer_index,
        precompile_layer_index,
        pa::COL_SEL_ECRECOVER,
        pa::COL_IS_REAL,
    )
}

/// Bind the ECRECOVER chain AIR's `(input[0..128], output[0..32])` byte
/// tuples to the [`crate::precompile_io_air`] byte windows. Wraps the
/// zkp-side parameterized builder with real IO-AIR offsets.
///
/// **Length clamp**: `precompile_io_air::MAX_INPUT_LENGTH = 64`, so
/// only the first 64 of ECRECOVER's 128 input bytes are bound. Full
/// coverage awaits the multi-row IO chunking AIR. The 32 output bytes
/// land in the SHA256 output channel (shape-only sink).
pub fn make_ecrecover_to_precompile_io_descriptor(
    ecrecover_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    // The zkp-side builder takes the full INPUT_LEN; we adapt to the
    // clamped IO width by constructing the descriptor directly to keep
    // A and B side lengths consistent.
    let input_clamp = ecr::INPUT_LEN.min(pio::MAX_INPUT_LENGTH);
    let output_clamp = ecr::OUTPUT_LEN.min(pio::SHA256_OUTPUT_LENGTH);

    let mut a_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    let mut b_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    for k in 0..input_clamp {
        a_columns.push(ecr::COL_INPUT_OFFSET + k);
        b_columns.push(pio::COL_INPUT_BYTES_OFFSET + k);
    }
    for k in 0..output_clamp {
        a_columns.push(ecr::COL_OUTPUT_OFFSET + k);
        b_columns.push(pio::COL_SHA256_OUTPUT_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "ecrecover_to_precompile_io_v1_clamped".into(),
        a_layer_index: ecrecover_layer_index,
        a_columns,
        a_selector_column: Some(ecr::COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: Some(pio::COL_IS_REAL),
    }
}

// ─── BN254 (callee 0x06 / 0x07) ───────────────────────────────────────

use metavm_zkp::bn254_precompile_air as bn254;

/// Bind the BN254 AIR's `(sel_ecadd, sel_ecmul, input_length,
/// gas_cost)` dispatch tuple to the [`crate::precompile_air`] row for
/// callees `0x06` (ECADD) and `0x07` (ECMUL). Output length (64) is
/// fixed and bound shape-only via selectors.
pub fn make_bn254_to_precompile_dispatch_descriptor(
    bn254_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = vec![
        bn254::COL_SEL_ECADD,
        bn254::COL_SEL_ECMUL,
        bn254::COL_INPUT_LENGTH,
        bn254::COL_GAS_COST,
    ];
    let b_columns: Vec<usize> = vec![
        pa::COL_SEL_ECADD,
        pa::COL_SEL_ECMUL,
        pa::COL_INPUT_LENGTH,
        pa::COL_GAS_COST,
    ];
    CrossAirLogUpDescriptor {
        label: "bn254_to_precompile_dispatch_v1".into(),
        a_layer_index: bn254_layer_index,
        a_columns,
        a_selector_column: Some(bn254::COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: Some(pa::COL_IS_REAL),
    }
}

/// Bind the BN254 AIR's `(input[0..128], output[0..64])` byte ranges to
/// the [`crate::precompile_io_air`] byte windows. Input is clamped to
/// the IO AIR's `MAX_INPUT_LENGTH = 64`; output is clamped to the
/// SHA256 output channel's 32 bytes.
pub fn make_bn254_to_precompile_io_descriptor(
    bn254_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let input_clamp = bn254::MAX_INPUT_LENGTH.min(pio::MAX_INPUT_LENGTH);
    let output_clamp = bn254::OUTPUT_LENGTH.min(pio::SHA256_OUTPUT_LENGTH);

    let mut a_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    let mut b_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    for k in 0..input_clamp {
        a_columns.push(bn254::COL_INPUT_OFFSET + k);
        b_columns.push(pio::COL_INPUT_BYTES_OFFSET + k);
    }
    for k in 0..output_clamp {
        a_columns.push(bn254::COL_OUTPUT_OFFSET + k);
        b_columns.push(pio::COL_SHA256_OUTPUT_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "bn254_to_precompile_io_v1_clamped".into(),
        a_layer_index: bn254_layer_index,
        a_columns,
        a_selector_column: Some(bn254::COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: Some(pio::COL_IS_REAL),
    }
}

// ─── BLAKE2F (callee 0x09) ────────────────────────────────────────────

use metavm_zkp::blake2f_precompile_air as blake2f;

/// Bind the BLAKE2F AIR's `(is_real, gas_cost)` dispatch tuple to the
/// corresponding [`crate::precompile_air`] row for callee `0x09`.
///
/// The zkp-side BLAKE2F AIR commits `gas_cost` directly as
/// `COL_GAS_COST` but does **not** commit `input_length` / `output_length`
/// columns (they're fixed at 213 / 64 by the layout). Length binding is
/// thus shape-only via the selector gating; the gas-cost column is bound
/// quantitatively.
pub fn make_blake2f_to_precompile_dispatch_descriptor(
    blake2f_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    // A side: (is_real anchor, gas_cost).
    let a_columns: Vec<usize> = vec![blake2f::COL_IS_REAL, blake2f::COL_GAS_COST];
    // B side: (sel_blake2f anchor on dispatched rows, gas_cost).
    let b_columns: Vec<usize> = vec![pa::COL_SEL_BLAKE2F, pa::COL_GAS_COST];
    CrossAirLogUpDescriptor {
        label: "blake2f_to_precompile_dispatch_v1".into(),
        a_layer_index: blake2f_layer_index,
        a_columns,
        a_selector_column: Some(blake2f::COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: Some(pa::COL_IS_REAL),
    }
}

/// Bind the BLAKE2F AIR's `(input[0..213], output[0..64])` byte ranges to
/// the [`crate::precompile_io_air`] byte windows.
///
/// **Length clamp**: `precompile_io_air::MAX_INPUT_LENGTH = 64`, so only
/// the first 64 of BLAKE2F's 213 input bytes (the `rounds + h_in` prefix)
/// are bound. Output is clamped to the SHA-256 output channel's 32 bytes
/// (BLAKE2F output is 64 bytes; second half awaits IO-AIR extension).
pub fn make_blake2f_to_precompile_io_descriptor(
    blake2f_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let input_clamp = blake2f::INPUT_LENGTH.min(pio::MAX_INPUT_LENGTH);
    let output_clamp = blake2f::OUTPUT_LENGTH.min(pio::SHA256_OUTPUT_LENGTH);

    let mut a_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    let mut b_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    for k in 0..input_clamp {
        a_columns.push(blake2f::COL_INPUT_OFFSET + k);
        b_columns.push(pio::COL_INPUT_BYTES_OFFSET + k);
    }
    for k in 0..output_clamp {
        a_columns.push(blake2f::COL_OUTPUT_OFFSET + k);
        // BLAKE2F output borrows the SHA-256 output channel until the IO
        // AIR adds a dedicated BLAKE2F output sink.
        b_columns.push(pio::COL_SHA256_OUTPUT_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "blake2f_to_precompile_io_v1_clamped".into(),
        a_layer_index: blake2f_layer_index,
        a_columns,
        a_selector_column: Some(blake2f::COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: Some(pio::COL_IS_REAL),
    }
}

// ─── MODEXP (callee 0x05) ─────────────────────────────────────────────

use metavm_zkp::modexp_precompile_air as modexp;

/// Bind the MODEXP AIR's `(is_real, gas_cost)` dispatch tuple to the
/// corresponding [`crate::precompile_air`] row for callee `0x05`.
///
/// MODEXP inputs are variable-length, so the zkp AIR does not commit a
/// single `input_length` column matching `precompile_air::COL_INPUT_LENGTH`
/// (the dispatch length there is the **call data** length). Length
/// binding is shape-only via the selectors; `gas_cost` is bound
/// quantitatively.
pub fn make_modexp_to_precompile_dispatch_descriptor(
    modexp_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = vec![modexp::COL_IS_REAL, modexp::COL_GAS_COST];
    let b_columns: Vec<usize> = vec![pa::COL_SEL_MODEXP, pa::COL_GAS_COST];
    CrossAirLogUpDescriptor {
        label: "modexp_to_precompile_dispatch_v1".into(),
        a_layer_index: modexp_layer_index,
        a_columns,
        a_selector_column: Some(modexp::COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: Some(pa::COL_IS_REAL),
    }
}

/// Bind the MODEXP AIR's input-header bytes
/// `input_header[0..INPUT_HEADER_LENGTH]` (the 96-byte
/// `(bsize, esize, msize)` prefix) and `output[0..MAX_OUTPUT_LENGTH]` to
/// the [`crate::precompile_io_air`] byte windows.
///
/// **Length clamp**: `precompile_io_air::MAX_INPUT_LENGTH = 64`, so only
/// the first 64 of the 96-byte input header are bound (covers the full
/// `bsize` size field plus half of `esize`). The variable-length
/// `b`/`e`/`m` payload bytes are **not** bound here; that awaits the
/// multi-row IO chunking AIR. Output is clamped to the SHA-256 output
/// channel's 32 bytes.
pub fn make_modexp_to_precompile_io_descriptor(
    modexp_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let input_clamp = modexp::INPUT_HEADER_LENGTH.min(pio::MAX_INPUT_LENGTH);
    let output_clamp = modexp::MAX_OUTPUT_LENGTH.min(pio::SHA256_OUTPUT_LENGTH);

    let mut a_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    let mut b_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    for k in 0..input_clamp {
        a_columns.push(modexp::COL_INPUT_HEADER_OFFSET + k);
        b_columns.push(pio::COL_INPUT_BYTES_OFFSET + k);
    }
    for k in 0..output_clamp {
        a_columns.push(modexp::COL_OUTPUT_OFFSET + k);
        b_columns.push(pio::COL_SHA256_OUTPUT_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "modexp_to_precompile_io_v1_clamped".into(),
        a_layer_index: modexp_layer_index,
        a_columns,
        a_selector_column: Some(modexp::COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: Some(pio::COL_IS_REAL),
    }
}

// ─── BN254 PAIRING (callee 0x08) ──────────────────────────────────────

use metavm_zkp::bn254_pairing_precompile_air as bn254_pairing;

/// Bind the BN254-pairing AIR's `(sel_ecpairing, gas_cost)` dispatch
/// tuple to the corresponding [`crate::precompile_air`] row for callee
/// `0x08`. Input length (a multiple of 192) varies with the number of
/// pairs; the per-row `gas_cost` column is bound quantitatively, while
/// the input-length / output-length binding is shape-only via the
/// selectors.
pub fn make_bn254_pairing_to_precompile_dispatch_descriptor(
    bn254_pairing_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = vec![
        bn254_pairing::COL_SEL_ECPAIRING,
        bn254_pairing::COL_GAS_COST,
    ];
    let b_columns: Vec<usize> = vec![pa::COL_SEL_ECPAIRING, pa::COL_GAS_COST];
    CrossAirLogUpDescriptor {
        label: "bn254_pairing_to_precompile_dispatch_v1".into(),
        a_layer_index: bn254_pairing_layer_index,
        a_columns,
        a_selector_column: Some(bn254_pairing::COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: Some(pa::COL_IS_REAL),
    }
}

/// Bind the BN254-pairing AIR's `(input[0..MAX_INPUT_LENGTH],
/// output[0..32])` byte ranges to the [`crate::precompile_io_air`] byte
/// windows.
///
/// **Length clamp**: `precompile_io_air::MAX_INPUT_LENGTH = 64`, so only
/// the first 64 input bytes of the up-to-`MAX_PAIRS * 192`-byte input
/// window are bound (covers the first G1 point `(x, y)`). Remaining pair
/// bytes await the multi-row IO chunking AIR. Output is exactly 32 bytes
/// (single big-endian word `0` or `1`) and lands fully in the SHA-256
/// output channel.
pub fn make_bn254_pairing_to_precompile_io_descriptor(
    bn254_pairing_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let input_clamp = bn254_pairing::MAX_INPUT_LENGTH.min(pio::MAX_INPUT_LENGTH);
    let output_clamp = bn254_pairing::OUTPUT_LENGTH.min(pio::SHA256_OUTPUT_LENGTH);

    let mut a_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    let mut b_columns: Vec<usize> = Vec::with_capacity(input_clamp + output_clamp);
    for k in 0..input_clamp {
        a_columns.push(bn254_pairing::COL_INPUT_OFFSET + k);
        b_columns.push(pio::COL_INPUT_BYTES_OFFSET + k);
    }
    for k in 0..output_clamp {
        a_columns.push(bn254_pairing::COL_OUTPUT_OFFSET + k);
        b_columns.push(pio::COL_SHA256_OUTPUT_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "bn254_pairing_to_precompile_io_v1_clamped".into(),
        a_layer_index: bn254_pairing_layer_index,
        a_columns,
        a_selector_column: Some(bn254_pairing::COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: Some(pio::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check that no EVM-side descriptor contains a
    /// `usize::MAX` sentinel anywhere in its A or B column lists.
    fn assert_no_sentinel(d: &CrossAirLogUpDescriptor) {
        for (i, c) in d.a_columns.iter().enumerate() {
            assert_ne!(
                *c,
                usize::MAX,
                "descriptor {:?} A-side col {} is sentinel",
                d.label,
                i
            );
        }
        for (i, c) in d.b_columns.iter().enumerate() {
            assert_ne!(
                *c,
                usize::MAX,
                "descriptor {:?} B-side col {} is sentinel",
                d.label,
                i
            );
        }
        if let Some(s) = d.a_selector_column {
            assert_ne!(s, usize::MAX, "descriptor {:?} A selector sentinel", d.label);
        }
        if let Some(s) = d.b_selector_column {
            assert_ne!(s, usize::MAX, "descriptor {:?} B selector sentinel", d.label);
        }
    }

    fn assert_b_in_precompile_air(d: &CrossAirLogUpDescriptor) {
        for c in &d.b_columns {
            assert!(
                *c < pa::NUM_COLUMNS,
                "descriptor {:?} B-col {} >= precompile_air NUM_COLUMNS {}",
                d.label,
                c,
                pa::NUM_COLUMNS
            );
        }
        if let Some(s) = d.b_selector_column {
            assert!(s < pa::NUM_COLUMNS);
        }
    }

    fn assert_b_in_precompile_io_air(d: &CrossAirLogUpDescriptor) {
        for c in &d.b_columns {
            assert!(
                *c < pio::NUM_COLUMNS,
                "descriptor {:?} B-col {} >= precompile_io_air NUM_COLUMNS {}",
                d.label,
                c,
                pio::NUM_COLUMNS
            );
        }
        if let Some(s) = d.b_selector_column {
            assert!(s < pio::NUM_COLUMNS);
        }
    }

    #[test]
    fn kzg_eval_descriptors_have_no_sentinels() {
        let d_disp = make_kzg_eval_to_precompile_dispatch_descriptor(0, 1);
        let d_io = make_kzg_eval_to_precompile_io_descriptor(0, 1);
        assert_no_sentinel(&d_disp);
        assert_no_sentinel(&d_io);
        // Dispatch is a 1-column gating tuple.
        assert_eq!(d_disp.a_columns.len(), 1);
        assert_eq!(d_disp.b_columns.len(), 1);
        // IO is clamped: 64 input + 32 output = 96.
        assert_eq!(d_io.a_columns.len(), 64 + 32);
        assert_eq!(d_io.b_columns.len(), 64 + 32);
        assert_b_in_precompile_air(&d_disp);
        assert_b_in_precompile_io_air(&d_io);
    }

    #[test]
    fn ripemd_descriptors_have_no_sentinels() {
        let d_disp = make_ripemd_to_precompile_dispatch_descriptor(0, 1);
        let d_io = make_ripemd_to_precompile_io_descriptor(0, 1);
        assert_no_sentinel(&d_disp);
        assert_no_sentinel(&d_io);
        // Dispatch: (is_real, input_length, gas_cost) — 3 cols.
        assert_eq!(d_disp.a_columns.len(), 3);
        assert_eq!(d_disp.b_columns.len(), 3);
        // IO: 64 input + 32 output = 96.
        assert_eq!(d_io.a_columns.len(), 64 + 32);
        assert_eq!(d_io.b_columns.len(), 64 + 32);
        assert_b_in_precompile_air(&d_disp);
        assert_b_in_precompile_io_air(&d_io);
    }

    #[test]
    fn ecrecover_descriptors_have_no_sentinels() {
        let d_disp = make_ecrecover_to_precompile_dispatch_descriptor(0, 1);
        let d_io = make_ecrecover_to_precompile_io_descriptor(0, 1);
        assert_no_sentinel(&d_disp);
        assert_no_sentinel(&d_io);
        // Dispatch: single anchor.
        assert_eq!(d_disp.a_columns.len(), 1);
        assert_eq!(d_disp.b_columns.len(), 1);
        // IO clamped: 64 input + 32 output = 96.
        assert_eq!(d_io.a_columns.len(), 64 + 32);
        assert_eq!(d_io.b_columns.len(), 64 + 32);
        assert_b_in_precompile_air(&d_disp);
        assert_b_in_precompile_io_air(&d_io);
    }

    #[test]
    fn bn254_descriptors_have_no_sentinels() {
        let d_disp = make_bn254_to_precompile_dispatch_descriptor(0, 1);
        let d_io = make_bn254_to_precompile_io_descriptor(0, 1);
        assert_no_sentinel(&d_disp);
        assert_no_sentinel(&d_io);
        // Dispatch: (sel_ecadd, sel_ecmul, input_length, gas_cost) — 4 cols.
        assert_eq!(d_disp.a_columns.len(), 4);
        assert_eq!(d_disp.b_columns.len(), 4);
        // IO clamped: 64 input + 32 output = 96.
        assert_eq!(d_io.a_columns.len(), 64 + 32);
        assert_eq!(d_io.b_columns.len(), 64 + 32);
        assert_b_in_precompile_air(&d_disp);
        assert_b_in_precompile_io_air(&d_io);
    }

    #[test]
    fn blake2f_dispatch_descriptor_has_no_sentinels() {
        let d_disp = make_blake2f_to_precompile_dispatch_descriptor(0, 1);
        assert_no_sentinel(&d_disp);
        // Dispatch: (is_real, gas_cost) — 2 cols.
        assert_eq!(d_disp.a_columns.len(), 2);
        assert_eq!(d_disp.b_columns.len(), 2);
        assert_b_in_precompile_air(&d_disp);
        // B-side selector must be the BLAKE2F sel column.
        assert_eq!(d_disp.b_columns[0], pa::COL_SEL_BLAKE2F);
    }

    #[test]
    fn blake2f_io_descriptor_has_no_sentinels() {
        let d_io = make_blake2f_to_precompile_io_descriptor(0, 1);
        assert_no_sentinel(&d_io);
        // IO clamped: 64 input (clamp from 213) + 32 output (clamp from 64) = 96.
        assert_eq!(d_io.a_columns.len(), 64 + 32);
        assert_eq!(d_io.b_columns.len(), 64 + 32);
        assert_b_in_precompile_io_air(&d_io);
    }

    #[test]
    fn modexp_dispatch_descriptor_has_no_sentinels() {
        let d_disp = make_modexp_to_precompile_dispatch_descriptor(0, 1);
        assert_no_sentinel(&d_disp);
        // Dispatch: (is_real, gas_cost) — 2 cols.
        assert_eq!(d_disp.a_columns.len(), 2);
        assert_eq!(d_disp.b_columns.len(), 2);
        assert_b_in_precompile_air(&d_disp);
        assert_eq!(d_disp.b_columns[0], pa::COL_SEL_MODEXP);
    }

    #[test]
    fn modexp_io_descriptor_has_no_sentinels() {
        let d_io = make_modexp_to_precompile_io_descriptor(0, 1);
        assert_no_sentinel(&d_io);
        // IO clamped: 64 input header bytes (clamp from 96) + 32 output (clamp
        // from 64) = 96.
        assert_eq!(d_io.a_columns.len(), 64 + 32);
        assert_eq!(d_io.b_columns.len(), 64 + 32);
        assert_b_in_precompile_io_air(&d_io);
    }

    #[test]
    fn bn254_pairing_dispatch_descriptor_has_no_sentinels() {
        let d_disp = make_bn254_pairing_to_precompile_dispatch_descriptor(0, 1);
        assert_no_sentinel(&d_disp);
        // Dispatch: (sel_ecpairing, gas_cost) — 2 cols.
        assert_eq!(d_disp.a_columns.len(), 2);
        assert_eq!(d_disp.b_columns.len(), 2);
        assert_b_in_precompile_air(&d_disp);
        assert_eq!(d_disp.b_columns[0], pa::COL_SEL_ECPAIRING);
    }

    #[test]
    fn bn254_pairing_io_descriptor_has_no_sentinels() {
        let d_io = make_bn254_pairing_to_precompile_io_descriptor(0, 1);
        assert_no_sentinel(&d_io);
        // IO clamped: 64 input bytes (clamp from MAX_INPUT_LENGTH) + 32 output
        // (the full 32-byte output) = 96.
        assert_eq!(d_io.a_columns.len(), 64 + 32);
        assert_eq!(d_io.b_columns.len(), 64 + 32);
        assert_b_in_precompile_io_air(&d_io);
    }

    #[test]
    fn layer_indices_pass_through() {
        // Layer indices propagate verbatim — important for the joint
        // prover orchestrator that chooses layer slots dynamically.
        let d = make_kzg_eval_to_precompile_dispatch_descriptor(7, 11);
        assert_eq!(d.a_layer_index, 7);
        assert_eq!(d.b_layer_index, 11);

        let d = make_bn254_to_precompile_io_descriptor(3, 5);
        assert_eq!(d.a_layer_index, 3);
        assert_eq!(d.b_layer_index, 5);
    }

    #[test]
    fn labels_are_non_stub() {
        // The EVM-side descriptors must NOT carry the zkp-side "_stub"
        // label, otherwise the joint-prover orchestrator may
        // accidentally skip them under future stub-filtering policy.
        for d in [
            make_kzg_eval_to_precompile_dispatch_descriptor(0, 1),
            make_kzg_eval_to_precompile_io_descriptor(0, 1),
            make_ripemd_to_precompile_dispatch_descriptor(0, 1),
            make_ripemd_to_precompile_io_descriptor(0, 1),
            make_ecrecover_to_precompile_dispatch_descriptor(0, 1),
            make_ecrecover_to_precompile_io_descriptor(0, 1),
            make_bn254_to_precompile_dispatch_descriptor(0, 1),
            make_bn254_to_precompile_io_descriptor(0, 1),
            make_blake2f_to_precompile_dispatch_descriptor(0, 1),
            make_blake2f_to_precompile_io_descriptor(0, 1),
            make_modexp_to_precompile_dispatch_descriptor(0, 1),
            make_modexp_to_precompile_io_descriptor(0, 1),
            make_bn254_pairing_to_precompile_dispatch_descriptor(0, 1),
            make_bn254_pairing_to_precompile_io_descriptor(0, 1),
        ] {
            assert!(
                !d.label.contains("_stub"),
                "descriptor label {:?} still marked as stub",
                d.label
            );
        }
    }
}
