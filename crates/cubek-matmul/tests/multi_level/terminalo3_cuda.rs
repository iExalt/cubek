//! CUDA-only regressions retained from the TerminalO3 fixture.
//!
//! These tests intentionally use the current public `launch_ref` and test
//! harness.  They are compile-only in the normal M0 qualification because the
//! CUDA device is not available on the qualification host; the `cuda-tests`
//! feature keeps them opt-in for a CUDA runner.

use cubecl::{
    Device,
    cuda::CudaDevice,
    frontend::Scalar,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides, shape},
};
use cubek_matmul::{
    definition::MatmulElems,
    launch::launch_ref,
    multi_level::{
        Strategy as MultiLevel,
        components::tile::TileMatmulKind,
        routines::batch::{
            double_buffering::DoubleBufferingArgs, ordered_double_buffering::OrderedSelectionArgs,
            simple::SimpleArgs,
        },
    },
    routine::BlueprintStrategy,
    strategy::Strategy,
};
use cubek_std::InputBinding;
use cubek_test_utils::{HostData, HostDataType, LayoutSpec, StridedLayout, TestInput};

use crate::harness::{f16_elems, f32_elems};

const LARGE_M: usize = 2_097_153;

fn client() -> Client {
    Device::Cuda(CudaDevice::default()).client()
}

fn parity(strategy: Strategy, m: usize, n: usize, k: usize, repeats: usize) {
    parity_with_layouts_and_dtype(
        strategy,
        m,
        n,
        k,
        StridedLayout::RowMajor,
        StridedLayout::RowMajor,
        f16_elems(),
        repeats,
    );
}

fn parity_with_layouts(
    strategy: Strategy,
    m: usize,
    n: usize,
    k: usize,
    lhs_layout: StridedLayout,
    rhs_layout: StridedLayout,
    repeats: usize,
) {
    parity_with_layouts_and_dtype(
        strategy,
        m,
        n,
        k,
        lhs_layout,
        rhs_layout,
        f16_elems(),
        repeats,
    );
}

/// Launch repeatedly with one input set and one output allocation. This keeps
/// the output-reuse and odd-tail regressions meaningful: rebuilding tensors for
/// each repetition cannot expose stale tile/barrier state on a second launch.
fn parity_with_layouts_and_dtype(
    strategy: Strategy,
    m: usize,
    n: usize,
    k: usize,
    lhs_layout: StridedLayout,
    rhs_layout: StridedLayout,
    elems: cubek_matmul::definition::MatmulGlobalElems,
    repeats: usize,
) {
    let client = client();
    let lhs = TestInput::builder(client.clone(), shape![m, k])
        .dtype(elems.lhs)
        .layout(LayoutSpec::from(lhs_layout))
        .custom(vec![1.0; m * k])
        .generate_without_host_data();
    let rhs = TestInput::builder(client.clone(), shape![k, n])
        .dtype(elems.rhs)
        .layout(LayoutSpec::from(rhs_layout))
        .custom(vec![1.0; k * n])
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(elems.out)
        .zeros()
        .generate_without_host_data();
    let lhs_binding = InputBinding::Normal(lhs.binding(), elems.lhs);
    let rhs_binding = InputBinding::Normal(rhs.binding(), elems.rhs);
    let mut dtypes = MatmulElems::from_globals(&elems);

    for repetition in 0..repeats {
        launch_ref(
            &strategy,
            &client,
            lhs_binding.clone(),
            rhs_binding.clone(),
            out.clone().binding(),
            &mut dtypes,
        )
        .unwrap_or_else(|error| {
            panic!("{strategy} failed on reusable-output launch {repetition}: {error:?}")
        });

        let actual = HostData::from_tensor_handle(&client, out.clone(), HostDataType::F32);
        for row in 0..m {
            for col in 0..n {
                let value = actual.get_f32(&[row, col]);
                assert_eq!(
                    value, k as f32,
                    "{strategy} reusable-output launch {repetition} produced {value} at ({row}, {col}), expected {k}"
                );
            }
        }
    }
}

fn parity_f32(strategy: Strategy, m: usize, n: usize, k: usize, repeats: usize) {
    parity_with_layouts_and_dtype(
        strategy,
        m,
        n,
        k,
        StridedLayout::RowMajor,
        StridedLayout::RowMajor,
        f32_elems(),
        repeats,
    );
}

#[test]
fn test_terminalo3_naive_large_m_axis_parity() {
    parity(MultiLevel::Naive.into(), LARGE_M, 1, 1, 1);
}

#[test]
fn test_terminalo3_cyclic_cmma_mma_output_reuse_parity() {
    for strategy in [
        MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: true,
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: true,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        MultiLevel::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(4),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(4),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        MultiLevel::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(8),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(8),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        MultiLevel::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: false,
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: false,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        MultiLevel::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: true,
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: true,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        MultiLevel::SpecializedCyclicCmma(BlueprintStrategy::Inferred(().into())).into(),
        MultiLevel::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())).into(),
    ] {
        parity_with_layouts(
            strategy,
            256,
            512,
            256,
            StridedLayout::RowMajor,
            StridedLayout::ColMajor,
            2,
        );
    }
}

#[test]
fn test_terminalo3_specialized_double_buffering_large_m_parity() {
    for n in [512, 264] {
        for strategy in [
            MultiLevel::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: true,
                tile_matmul: TileMatmulKind::Cmma,
            }))
            .into(),
            MultiLevel::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: true,
                tile_matmul: TileMatmulKind::Mma,
            }))
            .into(),
        ] {
            parity(strategy, 2048, n, 256, 2);
        }
    }
}

#[test]
fn test_terminalo3_specialized_double_buffering_medium_k_parity() {
    for (m, k) in [(1862, 1568), (1885, 1568), (1024, 1568), (1024, 2048)] {
        for strategy in [
            MultiLevel::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: true,
                tile_matmul: TileMatmulKind::Cmma,
            }))
            .into(),
            MultiLevel::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: true,
                tile_matmul: TileMatmulKind::Mma,
            }))
            .into(),
        ] {
            parity(strategy, m, 512, k, 2);
        }
    }
}

#[test]
fn test_terminalo3_specialized_double_buffering_odd_stage_tail_reuse_parity() {
    let strategy: Strategy =
        MultiLevel::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: true,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into();
    for _ in 0..8 {
        parity(strategy.clone(), 1862, 512, 1568, 2);
    }
}

#[test]
fn test_terminalo3_ordered_double_buffering_medium_k_parity() {
    for m in [1862, 1885] {
        for strategy in [
            MultiLevel::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                partition_k: Some(2),
                row_count: Some(8),
                rows_per_plane: Some(2),
                tile_matmul: TileMatmulKind::Cmma,
            }))
            .into(),
            MultiLevel::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                partition_k: Some(2),
                row_count: Some(8),
                rows_per_plane: Some(2),
                tile_matmul: TileMatmulKind::Mma,
            }))
            .into(),
        ] {
            parity(strategy, m, 512, 1568, 2);
        }
    }
}

#[test]
fn test_terminalo3_ordered_double_buffering_odd_stage_tail_reuse_parity() {
    let strategy: Strategy =
        MultiLevel::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(8),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into();
    for _ in 0..8 {
        parity(strategy.clone(), 1862, 512, 1568, 2);
    }
}

#[test]
fn test_terminalo3_specialized_cyclic_medium_k_parity() {
    for strategy in [
        MultiLevel::SpecializedCyclicCmma(BlueprintStrategy::Inferred(().into())).into(),
        MultiLevel::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())).into(),
    ] {
        parity(strategy, 1885, 512, 1568, 2);
    }
}

#[test]
fn test_terminalo3_specialized_cyclic_odd_stage_f32_parity() {
    for strategy in [
        MultiLevel::SpecializedCyclicCmma(BlueprintStrategy::Inferred(().into())).into(),
        MultiLevel::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())).into(),
    ] {
        parity_f32(strategy, 1885, 512, 1568, 1);
    }
}

#[test]
fn test_terminalo3_cyclic_large_m_axis_parity() {
    for strategy in [
        MultiLevel::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
    ] {
        parity_with_layouts_and_dtype(
            strategy,
            262_144,
            32,
            32,
            StridedLayout::RowMajor,
            StridedLayout::ColMajor,
            f16_elems(),
            2,
        );
    }
}

#[test]
fn test_terminalo3_tma_cmma_mma_output_reuse_parity() {
    for strategy in [
        MultiLevel::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        MultiLevel::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: true,
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: true,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        MultiLevel::SpecializedTmaCmma(BlueprintStrategy::Inferred(().into())).into(),
        MultiLevel::SpecializedTmaMma(BlueprintStrategy::Inferred(().into())).into(),
    ] {
        parity(strategy, 256, 512, 256, 2);
    }
}

fn launch_with_empty_tensors(
    strategy: &Strategy,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), cubek_matmul::definition::MatmulSetupError> {
    let client = client();
    let dtype = half::f16::elem_type_native();
    let lhs = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let rhs = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    launch_with_bindings(strategy, &client, lhs, rhs, out, dtype)
}

/// Create one-byte-sized logical tensors for setup-only validation. The TMA
/// selector must reject the oversized logical tile before any full allocation
/// or dispatch is attempted.
fn logical_tensor(client: &Client, shape: Shape) -> TensorHandle {
    let mut strides = vec![0; shape.len()];
    strides[shape.len() - 1] = 1;
    for index in (0..shape.len() - 1).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }

    TensorHandle::new(
        client.create_from_slice(half::f16::as_bytes(&[half::f16::ZERO])),
        shape,
        Strides::new(&strides),
        half::f16::elem_type_native(),
    )
}

fn launch_with_logical_tensors(
    strategy: &Strategy,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), cubek_matmul::definition::MatmulSetupError> {
    let client = client();
    let dtype = half::f16::elem_type_native();
    let lhs = logical_tensor(&client, shape![m, k]);
    let rhs = logical_tensor(&client, shape![k, n]);
    let out = logical_tensor(&client, shape![m, n]);
    launch_with_bindings(strategy, &client, lhs, rhs, out, dtype)
}

fn launch_with_bindings(
    strategy: &Strategy,
    client: &Client,
    lhs: TensorHandle,
    rhs: TensorHandle,
    out: TensorHandle,
    dtype: cubecl::ir::ElemType,
) -> Result<(), cubek_matmul::definition::MatmulSetupError> {
    let mut dtypes = MatmulElems::from_single_dtype(dtype);

    launch_ref(
        strategy,
        client,
        InputBinding::Normal(lhs.binding(), dtype),
        InputBinding::Normal(rhs.binding(), dtype),
        out.binding(),
        &mut dtypes,
    )
}

#[test]
fn test_terminalo3_tma_cmma_oversized_tile_rejected() {
    for strategy in [
        MultiLevel::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Cmma,
        }))
        .into(),
        MultiLevel::SpecializedTmaCmma(BlueprintStrategy::Inferred(().into())).into(),
    ] {
        let error = launch_with_logical_tensors(&strategy, 8192, 512, 2048)
            .expect_err("oversized TMA tile should be rejected during setup");
        let error = format!("{error:?}");
        assert!(
            error.contains("TMA tile shape")
                && (error.contains("1..=256") || error.contains("<= 256")),
            "{strategy} returned unexpected error: {error}",
        );
    }

    parity(
        MultiLevel::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Mma,
        }))
        .into(),
        256,
        512,
        256,
        2,
    );
}

#[test]
fn test_terminalo3_gemv_selector_parity() {
    for strategy in [
        MultiLevel::SimpleVecMat(Default::default()).into(),
        MultiLevel::DoubleVecMat(Default::default()).into(),
    ] {
        parity_with_layouts(
            strategy,
            1,
            256,
            256,
            StridedLayout::RowMajor,
            StridedLayout::ColMajor,
            2,
        );
    }

    parity(
        MultiLevel::GemvUnitPerpendicular(Default::default()).into(),
        1,
        256,
        256,
        2,
    );
}

#[test]
fn test_terminalo3_gemv_selectors_reject_general_matmul() {
    for strategy in [
        MultiLevel::SimpleVecMat(Default::default()).into(),
        MultiLevel::DoubleVecMat(Default::default()).into(),
        MultiLevel::GemvUnitPerpendicular(Default::default()).into(),
    ] {
        let error = launch_with_empty_tensors(&strategy, 256, 256, 512)
            .expect_err("GEMV selector accepted a general matmul");
        assert!(
            format!("{error:?}").contains("vec") || format!("{error:?}").contains("m = 1"),
            "{strategy} returned unexpected error: {error:?}",
        );
    }
}
