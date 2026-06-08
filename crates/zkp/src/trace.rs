use bls48581::bls48581::big;
use crate::field::{Scalar, CurveType};
use metavm_core::vm_traits::VmTrace;

/// Convert a u64 value to a BLS48-581 scalar field element (BIG).
///
/// Retained for backward compatibility. New code should use `Scalar::from_u64`.
pub fn u64_to_big(val: u64) -> big::BIG {
    // BIG::new_int takes an isize, but u64 values beyond isize::MAX need
    // byte-level construction. We always go through bytes for correctness.
    let mut buf = [0u8; big::MODBYTES];
    // BIG::frombytes reads big-endian. Place the 8 bytes of the u64 at the
    // end of the buffer (most-significant byte first within those 8 bytes).
    let start = big::MODBYTES - 8;
    buf[start] = (val >> 56) as u8;
    buf[start + 1] = (val >> 48) as u8;
    buf[start + 2] = (val >> 40) as u8;
    buf[start + 3] = (val >> 32) as u8;
    buf[start + 4] = (val >> 24) as u8;
    buf[start + 5] = (val >> 16) as u8;
    buf[start + 6] = (val >> 8) as u8;
    buf[start + 7] = val as u8;
    big::BIG::frombytes(&buf)
}

/// Return the smallest power of two that is >= n, with a minimum of 16.
/// BLS48-581 FFT only supports sizes {16, 32, 64, 128, 256}.
pub fn nearest_power_of_two(n: usize) -> usize {
    let mut power: usize = 16;
    while power < n {
        power <<= 1;
    }
    power
}

/// A polynomial represented in evaluation form (values at roots of unity).
#[derive(Clone)]
pub struct Polynomial {
    pub evaluations: Vec<Scalar>,
    pub degree: usize,
}

/// The trace converted to polynomials over a scalar field.
///
/// Columns are stored generically as a `Vec<Polynomial>`. The meaning of
/// each column index is defined by the VM-specific constraint system.
#[derive(Clone)]
pub struct TracePolynomials {
    pub columns: Vec<Polynomial>,
    /// Number of actual trace rows (before padding).
    pub num_rows: usize,
    /// Padded size (power of 2).
    pub padded_size: u64,
    /// Which curve these scalars target.
    pub curve: CurveType,
}

impl Polynomial {
    /// Create a polynomial from a slice of u64 values, targeting BLS48-581.
    /// Each value is converted to a Scalar and the result is padded to the next power of two.
    pub fn from_u64_vec(values: &[u64]) -> Self {
        Self::from_u64_vec_with_curve(values, CurveType::Bls48581)
    }

    /// Create a polynomial from a slice of u64 values for a specific curve.
    pub fn from_u64_vec_with_curve(values: &[u64], curve: CurveType) -> Self {
        let degree = values.len();
        let mut evaluations: Vec<Scalar> = values.iter()
            .map(|&v| Scalar::from_u64(v, curve))
            .collect();
        let padded_len = nearest_power_of_two(evaluations.len());
        evaluations.resize(padded_len, Scalar::zero(curve));
        Polynomial {
            evaluations,
            degree,
        }
    }

    /// Pad the evaluations vector with zeros until its length is a power of two.
    pub fn pad_to_power_of_two(&mut self) {
        let curve = if self.evaluations.is_empty() {
            CurveType::Bls48581
        } else {
            self.evaluations[0].curve_type()
        };
        let padded_len = nearest_power_of_two(self.evaluations.len());
        self.evaluations.resize(padded_len, Scalar::zero(curve));
    }

    /// Get evaluations as BLS48-581 BIG values (for backward compatibility with commitment.rs).
    pub fn as_bls48581_vec(&self) -> Vec<big::BIG> {
        self.evaluations.iter().map(|s| {
            match s {
                Scalar::Bls48581(b) => big::BIG::new_copy(b),
                _ => panic!("Expected BLS48-581 scalars"),
            }
        }).collect()
    }
}

impl TracePolynomials {
    /// Create TracePolynomials from any VM trace implementing VmTrace.
    ///
    /// Column 0 (step) is skipped; all remaining columns are collected generically.
    pub fn from_vm_trace(trace: &dyn VmTrace, curve: CurveType) -> Self {
        let raw_columns = trace.columns();
        let num_rows = trace.num_steps();
        let padded_size = nearest_power_of_two(num_rows) as u64;

        // Skip column 0 (step), collect remaining columns generically
        let columns: Vec<Polynomial> = raw_columns.iter()
            .skip(1)
            .map(|col| Polynomial::from_u64_vec_with_curve(col, curve))
            .collect();

        TracePolynomials {
            columns,
            num_rows,
            padded_size,
            curve,
        }
    }

    /// Create TracePolynomials directly from pre-built polynomials.
    pub fn from_polynomials(columns: Vec<Polynomial>, num_rows: usize, curve: CurveType) -> Self {
        let padded_size = if columns.is_empty() {
            nearest_power_of_two(num_rows) as u64
        } else {
            columns[0].evaluations.len() as u64
        };
        TracePolynomials {
            columns,
            num_rows,
            padded_size,
            curve,
        }
    }

    /// Return the number of actual execution steps (before padding).
    pub fn num_steps(&self) -> u64 {
        self.num_rows as u64
    }

    /// Return the padded domain size (power of 2).
    pub fn domain_size(&self) -> u64 {
        self.padded_size
    }

    /// Return references to all column evaluation vectors for commitment.
    pub fn columns(&self) -> Vec<&Vec<Scalar>> {
        self.columns.iter().map(|p| &p.evaluations).collect()
    }

    /// Fix selector column padding so that exactly one selector is 1 on padding rows.
    ///
    /// After polynomial construction, padding rows (indices `num_rows..padded_size`)
    /// have all-zero values including selectors. This violates the sum-to-one
    /// constraint. Call this method with the constraint system to set the designated
    /// "no-op" selector to 1 on all padding rows.
    pub fn fix_selector_padding(&mut self, constraints: &dyn crate::vm_constraints::VmConstraintSystem) {
        let sel_indices = constraints.selector_column_indices();
        if sel_indices.is_empty() {
            return;
        }
        let padding_col = match constraints.padding_selector_column() {
            Some(c) => c,
            None => return,
        };
        let one = Scalar::one(self.curve);
        let padded = self.padded_size as usize;
        // Set the padding selector to 1 on all padding rows
        for i in self.num_rows..padded {
            self.columns[padding_col].evaluations[i] = one.clone();
        }
    }

    /// Return columns as BLS48-581 BIG vectors (for backward compatibility with commitment/prover).
    pub fn columns_as_bls48581(&self) -> Vec<Vec<big::BIG>> {
        self.columns.iter().map(|p| {
            p.evaluations.iter().map(|s| match s {
                Scalar::Bls48581(b) => big::BIG::new_copy(b),
                _ => panic!("Expected BLS48-581 scalars"),
            }).collect()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_power_of_two() {
        // Minimum is 16 (BLS48-581 FFT lower bound)
        assert_eq!(nearest_power_of_two(0), 16);
        assert_eq!(nearest_power_of_two(1), 16);
        assert_eq!(nearest_power_of_two(2), 16);
        assert_eq!(nearest_power_of_two(3), 16);
        assert_eq!(nearest_power_of_two(4), 16);
        assert_eq!(nearest_power_of_two(5), 16);
        assert_eq!(nearest_power_of_two(8), 16);
        assert_eq!(nearest_power_of_two(9), 16);
        assert_eq!(nearest_power_of_two(16), 16);
        assert_eq!(nearest_power_of_two(17), 32);
        assert_eq!(nearest_power_of_two(32), 32);
        assert_eq!(nearest_power_of_two(33), 64);
        assert_eq!(nearest_power_of_two(256), 256);
    }

    #[test]
    fn test_u64_to_big() {
        let zero = u64_to_big(0);
        assert!(zero.iszilch());

        let one = u64_to_big(1);
        assert!(one.isunity());

        let val = u64_to_big(42);
        let expected = big::BIG::new_int(42);
        let mut diff = big::BIG::new_copy(&val);
        diff.sub(&expected);
        diff.norm();
        assert!(diff.iszilch());
    }

    #[test]
    fn test_polynomial_from_u64_vec() {
        let values = vec![0, 4, 8];
        let poly = Polynomial::from_u64_vec(&values);
        // 3 values should be padded to 16 (FFT minimum)
        assert_eq!(poly.evaluations.len(), 16);
        assert_eq!(poly.degree, 3);

        // Check the first value is zero
        assert!(poly.evaluations[0].is_zero());

        // Check the second value equals 4
        let four = Scalar::from_u64(4, CurveType::Bls48581);
        assert!(poly.evaluations[1].sub(&four).is_zero());

        // Check the padding values are zero
        for i in 3..16 {
            assert!(poly.evaluations[i].is_zero());
        }
    }

}
