//! ENV opcode context oracles.
//!
//! Verifies ADDRESS (0x30), ORIGIN (0x32), CALLER (0x33),
//! CALLVALUE (0x34), and GASPRICE (0x3A) outputs match the
//! transaction/call context.

use crate::trace::EvmTraceColumns;

pub struct TxContext {
    pub origin: [u8; 20],
    pub gas_price: u64,
    pub call_value: [u64; 4],
}

pub fn verify_origin(ctx: &TxContext, cols: &EvmTraceColumns) -> Result<(), String> {
    let n = cols.step.len();
    for r in 0..n {
        if cols.opcode[r] != 0x32 { continue; }
        let result_l0 = cols.output0[0][r];
        let expected_l0 = u64::from_be_bytes([
            ctx.origin[12], ctx.origin[13], ctx.origin[14], ctx.origin[15],
            ctx.origin[16], ctx.origin[17], ctx.origin[18], ctx.origin[19],
        ]);
        if result_l0 != expected_l0 {
            return Err(format!("step {}: ORIGIN mismatch", r));
        }
    }
    Ok(())
}

pub fn verify_gasprice(ctx: &TxContext, cols: &EvmTraceColumns) -> Result<(), String> {
    let n = cols.step.len();
    for r in 0..n {
        if cols.opcode[r] != 0x3A { continue; }
        let result = cols.output0[0][r];
        if result != ctx.gas_price {
            return Err(format!(
                "step {}: GASPRICE {} != expected {}",
                r, result, ctx.gas_price,
            ));
        }
    }
    Ok(())
}

pub fn verify_callvalue(ctx: &TxContext, cols: &EvmTraceColumns) -> Result<(), String> {
    let n = cols.step.len();
    for r in 0..n {
        if cols.opcode[r] != 0x34 { continue; }
        for limb in 0..4 {
            if cols.output0[limb][r] != ctx.call_value[limb] {
                return Err(format!(
                    "step {}: CALLVALUE limb {} mismatch: {} != {}",
                    r, limb, cols.output0[limb][r], ctx.call_value[limb],
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    #[test]
    fn origin_smoke() {
        // ORIGIN; STOP
        let bc = vec![0x32, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        // Default executor uses caller_addr = [0xCA; 20] as origin
        let ctx = TxContext {
            origin: [0xCA; 20],
            gas_price: 0,
            call_value: [0; 4],
        };
        verify_origin(&ctx, &cols).unwrap();
    }

    #[test]
    fn callvalue_zero() {
        // CALLVALUE; STOP
        let bc = vec![0x34, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let ctx = TxContext {
            origin: [0u8; 20],
            gas_price: 0,
            call_value: [0; 4],
        };
        verify_callvalue(&ctx, &cols).unwrap();
    }

    #[test]
    fn no_env_opcodes_passes() {
        let bc = vec![0x60, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let ctx = TxContext {
            origin: [0u8; 20],
            gas_price: 0,
            call_value: [0; 4],
        };
        verify_origin(&ctx, &cols).unwrap();
        verify_gasprice(&ctx, &cols).unwrap();
        verify_callvalue(&ctx, &cols).unwrap();
    }
}
