//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **call_family_air** on the
//! STATICCALL opcode (`0xFA`).
//!
//! Mirrors `crate::evm_main_address_joint_prove`. Bytecode:
//!
//!     PUSH1 0  PUSH1 0  PUSH1 0  PUSH1 0  PUSH1 0x04  PUSH3 0x100000
//!     STATICCALL  STOP
//!
//! i.e. `STATICCALL(gas=0x100000, addr=0x04, argOff=0, argLen=0,
//! retOff=0, retLen=0)`. revm pushes the (gas, addr, …) arguments onto
//! the stack so by the STATICCALL row the inspector observes:
//!
//!   - `COL_INPUT0_L0` = forwarded gas (top of stack)
//!   - `COL_INPUT1_L0` = callee address (second stack item)
//!   - `COL_SEL_STATICCALL = 1`
//!
//! The call_family_air gadget commits a single STATICCALL row with the
//! callee bound across all 4 LE u64 limbs and `is_call = 1`,
//! `sel_staticcall = 1`. We pull the actual callee bytes from the EVM
//! trace's `COL_INPUT1_L*` so the test is robust to whatever revm
//! observed (which should be 0x04 for this bytecode, but the gadget is
//! the source-of-truth for the value tuple either way).
//!
//! The descriptor binds **`COL_INPUT1_L0` ↔ `COL_CALLEE_L0`** as a
//! single-column tuple gated by the respective STATICCALL selectors
//! (`COL_SEL_STATICCALL` on the EVM side, `COL_SEL_STATICCALL` on the
//! gadget side). `joint_prove`'s current API requires single-column
//! tuples; the gadget AIR's row-local constraints (binary selectors,
//! 63/64 gas identity, STATICCALL value-zero per limb) pin the rest of
//! the witness.
//!
//! Slow tests are `#[ignore]`-gated for the same reason as the
//! ADDRESS template: BLS48-581 joint_prove takes O(30-60s) release.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::call_family_air::{
    COL_CALLEE_L0, COL_SEL_STATICCALL as GADGET_COL_SEL_STATICCALL,
};
use crate::trace::{COL_INPUT1_L0, COL_SEL_STATICCALL};

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_INPUT1_L0)` on STATICCALL rows to call_family_air's
/// `(COL_CALLEE_L0)` on real STATICCALL rows.
///
/// - A side: EVM main `COL_INPUT1_L0` gated by `COL_SEL_STATICCALL`.
/// - B side: call_family_air `COL_CALLEE_L0` gated by gadget's
///   `COL_SEL_STATICCALL`.
pub fn make_evm_call_to_call_family_descriptor(
    evm_layer_index: usize,
    call_family_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_call_to_call_family_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_INPUT1_L0],
        a_selector_column: Some(COL_SEL_STATICCALL),
        b_layer_index: call_family_layer_index,
        b_columns: vec![COL_CALLEE_L0],
        b_selector_column: Some(GADGET_COL_SEL_STATICCALL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::call_family_air::{
        build_trace_polynomials as build_cf_trace, CallEvent, CallFamilyConstraintSystem,
        CallFamilyWitness, KIND_STATICCALL,
    };
    use crate::constraints::EvmConstraintSystem;
    use crate::trace::{COL_INPUT1_L0, COL_INPUT1_L1, COL_INPUT1_L2, COL_INPUT1_L3, COL_SEL_STATICCALL};
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// Inner-callee address — a contract we pre-deploy with a single
    /// STOP opcode so the STATICCALL from the outer caller triggers a
    /// real new EVM frame (and an inner step) rather than executing a
    /// precompile inline. This matters for the EVM main shifted
    /// constraint `sel_staticcall * (depth_next - depth - 1) = 0`,
    /// which demands the NEXT trace row's frame_depth equal the
    /// STATICCALL row's depth + 1. Precompiles (0x01..0x09) execute
    /// inline in revm: the `call`/`call_end` inspector hooks fire
    /// synchronously between the outer STATICCALL row's `step_end`
    /// and the next outer row's `step`, so the next outer row is
    /// back at depth=0 — and the shifted constraint is violated. A
    /// real contract callee forces revm to actually `step` inside the
    /// new frame, so the row immediately after STATICCALL is at
    /// depth=1, satisfying the constraint.
    const INNER_CALLEE_ADDR: [u8; 20] = [
        0xBE, 0xEF, 0xBE, 0xEF, 0xBE, 0xEF, 0xBE, 0xEF, 0xBE, 0xEF,
        0xBE, 0xEF, 0xBE, 0xEF, 0xBE, 0xEF, 0xBE, 0xEF, 0xBE, 0xEF,
    ];

    /// Minimal STATICCALL bytecode targeting `INNER_CALLEE_ADDR` (a
    /// pre-deployed real-bytecode contract) with zero-length
    /// input/output. STATICCALL stack args (bottom → top): retLen,
    /// retOff, argLen, argOff, addr, gas. We PUSH in reverse order so
    /// by the STATICCALL opcode the stack top is `gas`.
    fn staticcall_bytecode() -> Vec<u8> {
        let mut code = Vec::new();
        // PUSH1 0x00  retLen
        code.extend_from_slice(&[0x60, 0x00]);
        // PUSH1 0x00  retOff
        code.extend_from_slice(&[0x60, 0x00]);
        // PUSH1 0x00  argLen
        code.extend_from_slice(&[0x60, 0x00]);
        // PUSH1 0x00  argOff
        code.extend_from_slice(&[0x60, 0x00]);
        // PUSH20 INNER_CALLEE_ADDR  addr
        code.push(0x73);
        code.extend_from_slice(&INNER_CALLEE_ADDR);
        // PUSH3 0x100000  gas (~1M)
        code.extend_from_slice(&[0x62, 0x10, 0x00, 0x00]);
        // STATICCALL
        code.push(0xFA);
        // STOP
        code.push(0x00);
        code
    }

    /// Custom executor that pre-deploys BOTH the outer caller contract
    /// (running `staticcall_bytecode`) at `[0x42;20]` AND an inner
    /// callee contract (a single STOP opcode) at `INNER_CALLEE_ADDR`.
    /// This ensures the STATICCALL from the outer contract enters a
    /// real EVM frame (not a precompile), so revm fires `step` inside
    /// the new frame and the EVM main shifted constraint
    /// `sel_staticcall * (depth_next - depth - 1) = 0` is satisfied.
    fn execute_outer_staticcalls_inner() -> crate::trace::EvmTraceColumns {
        use revm::context::{Context, TxEnv};
        use revm::database::{CacheDB, EmptyDB};
        use revm::handler::{MainBuilder, MainContext};
        use revm::primitives::{Address, Bytes, TxKind, U256};
        use revm::state::{AccountInfo, Bytecode};
        use revm::InspectEvm;

        let outer_addr = Address::from([0x42u8; 20]);
        let inner_addr = Address::from(INNER_CALLEE_ADDR);
        let caller_addr = Address::from([0xCAu8; 20]);

        let outer_bytecode = staticcall_bytecode();
        let inner_bytecode = vec![0x00u8]; // STOP

        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            outer_addr,
            AccountInfo {
                balance: U256::ZERO,
                nonce: 0,
                code_hash: Default::default(),
                account_id: None,
                code: Some(Bytecode::new_legacy(Bytes::copy_from_slice(&outer_bytecode))),
            },
        );
        db.insert_account_info(
            inner_addr,
            AccountInfo {
                balance: U256::ZERO,
                nonce: 0,
                code_hash: Default::default(),
                account_id: None,
                code: Some(Bytecode::new_legacy(Bytes::copy_from_slice(&inner_bytecode))),
            },
        );
        db.insert_account_info(
            caller_addr,
            AccountInfo {
                balance: U256::from(1_000_000_000_000_000_000u64),
                nonce: 0,
                code_hash: Default::default(),
                account_id: None,
                code: None,
            },
        );

        let inspector = crate::inspector::TracingInspector::new();
        let ctx = Context::mainnet().with_db(db);
        let mut evm = ctx.build_mainnet_with_inspector(inspector);
        let tx = TxEnv::builder()
            .caller(caller_addr)
            .kind(TxKind::Call(outer_addr))
            .gas_limit(10_000_000)
            .gas_price(0)
            .value(U256::ZERO)
            .data(Bytes::new())
            .nonce(0)
            .build_fill();
        let result = evm
            .inspect_tx(tx)
            .expect("outer→inner STATICCALL transaction must succeed");
        eprintln!(
            "[evm] outer→inner STATICCALL gas_used={}",
            result.result.gas_used()
        );
        evm.into_inspector().trace
    }

    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let cols = execute_outer_staticcalls_inner();
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Extract the 4 LE u64 callee-address limbs from the EVM trace's
    /// STATICCALL row (`COL_INPUT1_L0..L3`). Robust to whatever revm
    /// observed.
    fn extract_staticcall_callee(
        evm_polys: &TracePolynomials,
    ) -> Result<[u64; 4], &'static str> {
        let sel_col = &evm_polys.columns[COL_SEL_STATICCALL].evaluations;
        let curve = evm_polys.curve;
        let one_bytes = Scalar::one(curve).to_bytes();
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                let mut limbs = [0u64; 4];
                for (i, col) in [
                    COL_INPUT1_L0, COL_INPUT1_L1, COL_INPUT1_L2, COL_INPUT1_L3,
                ]
                .iter()
                .enumerate()
                {
                    let be = evm_polys.columns[*col].evaluations[r].to_bytes();
                    let mut tail = [0u8; 8];
                    tail.copy_from_slice(&be[be.len() - 8..]);
                    limbs[i] = u64::from_be_bytes(tail);
                }
                return Ok(limbs);
            }
        }
        Err("EVM main trace must contain at least one STATICCALL row")
    }

    /// Build a 1-row call_family_air witness for a STATICCALL event with
    /// callee read from the EVM trace.
    fn build_call_family_trace(
        evm_polys: &TracePolynomials,
        curve: CurveType,
    ) -> TracePolynomials {
        let callee = extract_staticcall_callee(evm_polys)
            .expect("EVM main trace must contain at least one STATICCALL row");
        // Honest 63/64 forwarding: gas_in * 63 = gas_forwarded * 64 + 0.
        // 64_000 * 63 = 63_000 * 64 = 4_032_000 → remainder 0.
        let event = CallEvent {
            call_op: KIND_STATICCALL,
            caller: [0; 4],
            callee,
            value: [0; 4], // STATICCALL value must be zero
            gas_in: 64_000,
            gas_forwarded: 63_000,
            gas_returned: 0,
            depth_pre: 1,
            depth_post: 2,
            is_call: true,
            is_return: false,
            is_static: true,
        };
        let w = CallFamilyWitness::from_events(vec![event]);
        build_cf_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and the EVM trace really contains
    /// a STATICCALL row whose `COL_INPUT1_L0` matches the gadget row's
    /// `COL_CALLEE_L0`.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_call_to_call_family_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_call_to_call_family_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_ne!(
            d.a_layer_index, d.b_layer_index,
            "linkage layers must differ for cross-AIR LogUp"
        );
        assert_eq!(d.a_columns.len(), d.b_columns.len());
        assert_eq!(
            d.a_columns.len(),
            1,
            "joint_prove currently requires single-column tuples"
        );
        assert_eq!(d.a_columns[0], COL_INPUT1_L0);
        assert_eq!(d.b_columns[0], COL_CALLEE_L0);
        assert_eq!(d.a_selector_column, Some(COL_SEL_STATICCALL));
        assert_eq!(d.b_selector_column, Some(GADGET_COL_SEL_STATICCALL));

        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let cf_polys = build_call_family_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let cf_cs = CallFamilyConstraintSystem::new(cf_polys.num_rows);

        // Confirm the EVM trace contains a STATICCALL row.
        let sel_col = &evm_polys.columns[COL_SEL_STATICCALL].evaluations;
        let in1_l0_col = &evm_polys.columns[COL_INPUT1_L0].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found = None;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                found = Some(in1_l0_col[r].to_bytes());
                break;
            }
        }
        let evm_l0_bytes =
            found.expect("EVM main trace must contain at least one STATICCALL row");

        // Gadget side: row 0 must mirror the same callee limb-0.
        let gadget_l0 = &cf_polys.columns[COL_CALLEE_L0].evaluations;
        assert_eq!(
            gadget_l0[0].to_bytes(),
            evm_l0_bytes,
            "gadget callee_l0 row 0 must equal EVM main STATICCALL row's INPUT1_L0",
        );

        // Gadget side STATICCALL selector must fire on the event row.
        let gadget_sel = &cf_polys.columns[GADGET_COL_SEL_STATICCALL].evaluations;
        assert_eq!(
            gadget_sel[0].to_bytes(),
            one_bytes,
            "gadget sel_staticcall must fire on the STATICCALL event row",
        );

        // Sanity: the two (trace, cs) pairs assemble into the
        // joint_prove input shape.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&cf_polys, &cf_cs)];
        let linkages = vec![make_evm_call_to_call_family_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);

        assert_eq!(KIND_STATICCALL, 3);
    }

    /// Fast diagnostic: ensure the per-AIR row-local constraint
    /// bodies all evaluate to zero on the cf witness AND that the EVM
    /// trace at the STATICCALL row has sensible columns. This catches
    /// witness construction bugs before the slow joint_prove runs.
    #[test]
    fn diag_cf_constraints_zero_on_staticcall_witness() {
        use metavm_zkp::vm_constraints::VmConstraintSystem;
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let cf_polys = build_call_family_trace(&evm_polys, curve);
        let cf_cs = CallFamilyConstraintSystem::new(cf_polys.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            cf_polys.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cf_cs.evaluate_on_domain(&cr, cf_polys.num_rows);
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "call_family_air row-local constraint {i} row {r} nonzero on STATICCALL witness",
                );
            }
        }
        // Diagnostics: dump cf trace at row 0 (last 8 bytes — BE u64 tail)
        let last8 = |b: Vec<u8>| -> Vec<u8> { b[b.len() - 8..].to_vec() };
        eprintln!(
            "[diag] cf row0: is_real={:?} is_call={:?} sel_staticcall={:?} callee_l0={:?}",
            last8(cf_polys.columns[crate::call_family_air::COL_IS_REAL].evaluations[0].to_bytes()),
            last8(cf_polys.columns[crate::call_family_air::COL_IS_CALL].evaluations[0].to_bytes()),
            last8(cf_polys.columns[GADGET_COL_SEL_STATICCALL].evaluations[0].to_bytes()),
            last8(cf_polys.columns[COL_CALLEE_L0].evaluations[0].to_bytes()),
        );
        // Also dump EVM row at STATICCALL
        let sel_col = &evm_polys.columns[COL_SEL_STATICCALL].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut staticcall_row = None;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                staticcall_row = Some(r);
                break;
            }
        }
        let r = staticcall_row.expect("STATICCALL row");
        eprintln!(
            "[diag] EVM staticcall row {}: SEL_STATICCALL={:?} INPUT1_L0={:?}",
            r,
            last8(evm_polys.columns[COL_SEL_STATICCALL].evaluations[r].to_bytes()),
            last8(evm_polys.columns[COL_INPUT1_L0].evaluations[r].to_bytes()),
        );
        eprintln!(
            "[diag] cf num_rows={} padded_size={}",
            cf_polys.num_rows, cf_polys.padded_size
        );
        eprintln!(
            "[diag] evm num_rows={} padded_size={}",
            evm_polys.num_rows, evm_polys.padded_size
        );
    }

    /// Fast diagnostic: evaluate EVM main row-local constraint bodies
    /// over the inflated trace (target_padded=256) and dump any
    /// nonzero locations. This is the cheapest way to discover whether
    /// the STATICCALL bytecode produces a trace that violates an EVM
    /// main AIR constraint — which would cause joint_verify to reject
    /// the EVM-side per-AIR proof.
    #[test]
    fn diag_evm_main_constraints_zero_on_staticcall_trace() {
        use metavm_zkp::vm_constraints::VmConstraintSystem;
        let curve = CurveType::Bls48581;
        let mut evm_polys = build_evm_main_trace(curve);
        // Inflate to 256 like joint_prove does (EVM has LogUp → 256).
        let zero = Scalar::zero(curve);
        for poly in evm_polys.columns.iter_mut() {
            poly.evaluations.resize(256, zero.clone());
        }
        let old_padded = evm_polys.padded_size;
        evm_polys.padded_size = 256;
        let evm_cs = EvmConstraintSystem::new();
        // Apply fix_trace_padding so we mirror what joint_prove sees.
        let num_rows = evm_polys.num_rows;
        let padded = evm_polys.padded_size as usize;
        let mut col_evals: Vec<Vec<Scalar>> = evm_polys
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        evm_cs.fix_trace_padding(&mut col_evals, num_rows, padded);
        for (poly, evals) in evm_polys.columns.iter_mut().zip(col_evals.into_iter()) {
            poly.evaluations = evals;
        }
        let cr: Vec<&Vec<Scalar>> =
            evm_polys.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = evm_cs.evaluate_on_domain(&cr, num_rows);
        let mut violations: Vec<(usize, usize)> = Vec::new();
        let labels = evm_cs.constraint_labels();
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                if !v.is_zero() {
                    violations.push((i, r));
                    if violations.len() <= 20 {
                        let lbl = labels.get(i).map(|s| s.as_str()).unwrap_or("?");
                        eprintln!("[diag] EVM main row-local constraint {i} ({lbl}) nonzero at row {r}");
                    }
                }
            }
        }
        eprintln!(
            "[diag] EVM main row-local constraints: {} violations total (was padded {old_padded} → {padded})",
            violations.len()
        );
        assert!(
            violations.is_empty(),
            "EVM main row-local constraints violated on STATICCALL trace ({} violations)",
            violations.len()
        );
    }

    /// Fast diagnostic: confirm the STATICCALL row has `depth_next ==
    /// depth + 1` (the EVM main shifted constraint demands this for
    /// every `sel_staticcall=1` row). If this fails, joint_prove cannot
    /// succeed because the EVM main shifted constraint polynomial is
    /// nonzero on the wrap-row test.
    #[test]
    fn diag_staticcall_depth_next_eq_depth_plus_one() {
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let sel_col = &evm_polys.columns[COL_SEL_STATICCALL].evaluations;
        let depth_col =
            &evm_polys.columns[crate::trace::COL_FRAME_DEPTH].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut sc_rows: Vec<usize> = Vec::new();
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                sc_rows.push(r);
            }
        }
        assert!(!sc_rows.is_empty(), "no STATICCALL row found in EVM trace");
        for r in sc_rows {
            let depth = &depth_col[r];
            let depth_next = if r + 1 < depth_col.len() {
                &depth_col[r + 1]
            } else {
                &depth_col[0]
            };
            let one = Scalar::one(curve);
            // Constraint: depth_next - depth - 1 == 0
            let body = depth_next.sub(depth).sub(&one);
            eprintln!(
                "[diag] STATICCALL row {} depth={:?} depth_next={:?} body={:?}",
                r,
                &depth.to_bytes()[depth.to_bytes().len() - 8..],
                &depth_next.to_bytes()[depth_next.to_bytes().len() - 8..],
                &body.to_bytes()[body.to_bytes().len() - 8..],
            );
            assert!(
                body.is_zero(),
                "STATICCALL row {} shifted constraint depth_next == depth + 1 violated",
                r
            );
        }
    }

    /// Fast diagnostic: dump key EVM trace columns row-by-row so we can
    /// manually inspect the shifted-constraint violations.
    ///
    /// **Task #298 finding (2026-06-05)**: `honest_joint_verify_diagnostic`
    /// fails release at `PerAirVerifyFailed { air_index: 0 }` with
    /// `Q(z)*Z(z) != C(z)` from the EVM main per-AIR verify. All
    /// row-local constraints pass (`diag_evm_main_constraints_zero_*`)
    /// so the violator is a **shifted constraint**.
    ///
    /// Root cause: the EVM main shifted constraint
    ///     `sel_staticcall * (frame_callee_next - INPUT1) = 0`
    /// (and the analogous constraints for CALL/CALLCODE/DELEGATECALL/
    /// CREATE/CREATE2) compares two columns that use **incompatible
    /// address encodings**:
    ///
    /// - `INPUT1_L0..3` is populated by `safe_peek(stack, 1)` → returns
    ///   the EVM stack's U256 `as_limbs()` (LE-limbs of the integer
    ///   value of the address). For inner `[0xBE,0xEF,...]×10`
    ///   (PUSH20), limb 0 = low 64 bits of the integer = `0xBEEFBEEFBEEFBEEF`.
    /// - `frame_callee_l0..3` is populated by
    ///   `address_to_limbs(addr)` → reads bytes 0..8 of the 20-byte
    ///   address as `u64::from_le_bytes`. For the same address, limb 0
    ///   = `0xEFBEEFBEEFBEEFBE`.
    ///
    /// These differ by a byte rotation. The constraint demands equality
    /// → it can never hold honestly on any non-zero address.
    ///
    /// This is **pre-existing**: no existing prove+verify test runs a
    /// CALL-family bytecode through the EVM main constraint system
    /// (`test_evm_prove_verify_roundtrip` uses `make_add_trace`). The
    /// failure surfaced once #256/#291 fixed the depth/wrap-factor
    /// issues and the constraint pipeline started actually exercising
    /// these shifted bodies.
    ///
    /// **Deferred fix options** (out of #298 scope):
    /// 1. Inspector: detect CALL-family opcodes in `step` and re-encode
    ///    `input1` to `address_to_limbs(low20bytes(stack_top1))` for
    ///    those rows only. Minimal blast radius — other consumers of
    ///    INPUT1 (BYTE/JUMPI/SSTORE/...) keep U256-numeric form.
    /// 2. Add a new column `frame_callee_input_hint` written by the
    ///    inspector in the `address_to_limbs` convention; rewrite all
    ///    CALL-family shifted constraints to bind that column instead
    ///    of INPUT1.
    /// 3. Globally change `address_to_limbs` to U256-numeric limbs —
    ///    cascades through storage_access_air, address_keccak_air,
    ///    account_state_air, keccak_extract, and at least two CREATE
    ///    RLP AIRs. Not recommended.
    ///
    /// The dump test below confirms the encoding mismatch directly:
    /// row 6 (STATICCALL) has INPUT1_L0 = 13758425323549998831
    /// (`0xBEEFBEEFBEEFBEEF`) while row 7 has F_CALLEE_L0 =
    /// 17275508823984893886 (`0xEFBEEFBEEFBEEFBE`).
    #[test]
    fn diag_dump_evm_trace_columns_staticcall() {
        let curve = CurveType::Bls48581;
        let mut evm_polys = build_evm_main_trace(curve);
        let zero = Scalar::zero(curve);
        for poly in evm_polys.columns.iter_mut() {
            poly.evaluations.resize(256, zero.clone());
        }
        evm_polys.padded_size = 256;
        let evm_cs = EvmConstraintSystem::new();
        let num_rows = evm_polys.num_rows;
        let padded = evm_polys.padded_size as usize;
        let mut col_evals: Vec<Vec<Scalar>> = evm_polys
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        use metavm_zkp::vm_constraints::VmConstraintSystem;
        evm_cs.fix_trace_padding(&mut col_evals, num_rows, padded);

        let last8 = |s: &Scalar| -> u64 {
            let b = s.to_bytes();
            let mut t = [0u8; 8];
            t.copy_from_slice(&b[b.len() - 8..]);
            u64::from_be_bytes(t)
        };

        eprintln!("[diag] num_rows={} padded={}", num_rows, padded);
        let dump_cols = [
            ("PC", crate::trace::COL_PC),
            ("NEXT_PC", crate::trace::COL_NEXT_PC),
            ("opcode", crate::trace::COL_OPCODE),
            ("F_DEPTH", crate::trace::COL_FRAME_DEPTH),
            ("F_STATIC", crate::trace::COL_FRAME_STATIC),
            ("F_RET_PC", crate::trace::COL_FRAME_RETURN_PC),
            ("F_CALLEE_L0", crate::trace::COL_FRAME_CALLEE_L0),
            ("F_CALLER_L0", crate::trace::COL_FRAME_CALLER_L0),
            ("INPUT1_L0", COL_INPUT1_L0),
            ("INPUT0_L0", crate::trace::COL_INPUT0_L0),
            ("sel_static", COL_SEL_STATICCALL),
            ("sel_push", crate::trace::COL_SEL_CALL_PUSH_FRAME),
            ("sel_ret", crate::trace::COL_SEL_CALL_RETURN),
            ("sel_stop_pop", crate::trace::COL_SEL_STOP_POP),
            ("sel_stop", crate::trace::COL_SEL_STOP),
        ];
        // Look at first 14 rows (real + a couple padding).
        for r in 0..14.min(padded) {
            eprint!("[row {:2}]", r);
            for (n, c) in &dump_cols {
                eprint!(" {}={}", n, last8(&col_evals[*c][r]));
            }
            eprintln!();
        }
    }

    /// Fast diagnostic: standalone prove+verify on the call_family_air
    /// witness only (no joint_prove machinery). If this fails the joint
    /// also will, and the bug is on the gadget side.
    #[test]
    #[ignore = "slow: standalone prove+verify of call_family_air on the STATICCALL witness"]
    fn diag_cf_standalone_prove_verify() {
        use metavm_zkp::prover::prove_with_scheme;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use metavm_zkp::verifier::verify_with_scheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let evm_polys = build_evm_main_trace(curve);
        let cf_polys = build_call_family_trace(&evm_polys, curve);
        let cf_cs = CallFamilyConstraintSystem::new(cf_polys.num_rows);
        let proof = prove_with_scheme(&cf_polys, &cf_cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cf_cs, &scheme, curve),
            "call_family_air standalone prove+verify must succeed on STATICCALL witness",
        );
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace (running the STATICCALL bytecode) and a 1-row
    /// call_family_air witness for the STATICCALL event.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + call_family_air STATICCALL (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let cf_polys = build_call_family_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let cf_cs = CallFamilyConstraintSystem::new(cf_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&cf_polys, &cf_cs)];
        let linkages = vec![make_evm_call_to_call_family_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + call_family STATICCALL) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's STATICCALL-row INPUT1_L0 multiset \
             equals B's real-row callee_l0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &cf_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ call_family STATICCALL linkage",
        );
    }

    /// Diagnostic: run joint_prove + joint_verify_diagnostic to identify
    /// the first failing check (per-AIR verify, linkage SNARK, cross-trace
    /// binding, closure-wrap, etc.).
    #[test]
    #[ignore = "slow: joint_prove + joint_verify_diagnostic STATICCALL"]
    fn honest_joint_verify_diagnostic() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify_diagnostic, JointVerifyFailure};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let cf_polys = build_call_family_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let cf_cs = CallFamilyConstraintSystem::new(cf_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&cf_polys, &cf_cs)];
        let linkages = vec![make_evm_call_to_call_family_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed");

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &cf_cs];
        let diag = joint_verify_diagnostic(
            &proofs, &cs_refs, &linkages, &ext, &scheme, curve,
        );
        eprintln!("[diag] joint_verify_diagnostic = {:?}", diag);
        assert_eq!(
            diag,
            JointVerifyFailure::Ok,
            "joint_verify_diagnostic must return Ok; got {:?}",
            diag
        );
    }

    /// Tampered envelope: corrupt `closure_a` so the joint verifier
    /// rejects.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_joint_verify_true"]
    fn tampered_joint_verify_false() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let cf_polys = build_call_family_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let cf_cs = CallFamilyConstraintSystem::new(cf_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&cf_polys, &cf_cs)];
        let linkages = vec![make_evm_call_to_call_family_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &cf_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched linkage closures",
        );
    }
}
