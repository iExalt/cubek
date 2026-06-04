/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;

pub mod cpu_gemm;
pub mod gemm;
pub mod gemv_unit_perpendicular;

/// The cooperative, tiled `BatchMatmulRoutine` family sharing one launch hub.
pub mod batch;

mod base;
mod selector;

use crate::definition::{MatmulProblem, MatmulSetupError};

pub use base::*;
pub use selector::*;

fn validate_vecmat_problem(problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
    if problem.m != 1 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "m must equal 1 to qualify as a vecmat problem, got {}",
            problem.m
        ))));
    }

    Ok(())
}
