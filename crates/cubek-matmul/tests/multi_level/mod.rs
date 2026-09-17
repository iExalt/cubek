//! Tests for the routines written on the batch/global/stage/tile levels.

mod basic;
mod bias;

#[cfg(feature = "cuda-tests")]
mod terminalo3_cuda;

#[cfg(feature = "benchmarks")]
mod bench_catalog;
#[cfg(feature = "benchmarks")]
mod comparison;
#[cfg(feature = "extended")]
mod extended;
#[cfg(feature = "full")]
mod full;
