use crate::field::Scalar;

/// A constraint is a polynomial that must vanish on the trace domain.
/// C(x) = 0 for all x in the evaluation domain.
#[derive(Clone)]
pub struct Constraint {
    pub label: String,
    pub degree: usize,
}

/// Result of evaluating constraints on a trace.
pub struct ConstraintEvaluation {
    /// For each constraint, the polynomial values. Should be all zeros if trace is valid.
    pub evaluations: Vec<Vec<Scalar>>,
}

impl ConstraintEvaluation {
    /// Return the number of evaluation points (rows) per constraint.
    pub fn num_evaluations(&self) -> usize {
        self.evaluations.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Return references to each constraint's evaluation column.
    pub fn columns(&self) -> &[Vec<Scalar>] {
        &self.evaluations
    }
}

/// Combine all constraint evaluation columns into a single constraint
/// polynomial using a random linear combination with challenge `alpha`.
pub fn build_constraint_polynomial(
    evaluations: &[Vec<Scalar>],
    alpha: &Scalar,
) -> Vec<Scalar> {
    let curve = alpha.curve_type();

    if evaluations.is_empty() {
        return Vec::new();
    }

    let n = evaluations[0].len();
    let mut result = vec![Scalar::zero(curve); n];

    let mut alpha_power = Scalar::one(curve);
    for constraint_evals in evaluations {
        for j in 0..n.min(constraint_evals.len()) {
            let term = alpha_power.mul(&constraint_evals[j]);
            result[j] = result[j].add(&term);
        }
        alpha_power = alpha_power.mul(alpha);
    }

    result
}
