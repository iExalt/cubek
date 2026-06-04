//! CUDA parity regressions extracted from TerminalO3 autotune artifacts.

use cubecl::{
    CubeElement, Runtime,
    client::ComputeClient,
    cuda::CudaRuntime,
    prelude::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides, shape},
};
use cubek_matmul::{
    components::tile::TileMatmulKind,
    definition::{MatmulElems, MatmulGlobalElems},
    launch::{Strategy, launch_ref},
    routines::{
        BlueprintStrategy, double_buffering::DoubleBufferingArgs,
        ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
    },
};
use cubek_std::InputBinding;
use half::f16;

const LARGE_M: usize = 2_097_153;

fn f16_dtypes() -> MatmulElems {
    let f16 = f16::as_type_native_unchecked().storage_type();
    MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: f16,
        rhs: f16,
        out: f16,
    })
}

fn filled_tensor(
    client: &ComputeClient<CudaRuntime>,
    shape: Shape,
    value: f32,
) -> TensorHandle<CudaRuntime> {
    let num_elements = shape.iter().product();
    let values = vec![f16::from_f32(value); num_elements];
    let layout = client.create_tensor_from_slice(f16::as_bytes(&values), shape.clone(), 2);
    TensorHandle::new(
        layout.memory,
        shape,
        layout.strides,
        f16::as_type_native_unchecked(),
    )
}

fn read_f16(client: &ComputeClient<CudaRuntime>, output: TensorHandle<CudaRuntime>) -> Vec<f32> {
    let bytes = client.read_one_unchecked_tensor(output.into_copy_descriptor());
    f16::from_bytes(&bytes)
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

fn transposed_filled_tensor(
    client: &ComputeClient<CudaRuntime>,
    shape: Shape,
    value: f32,
) -> TensorHandle<CudaRuntime> {
    let num_elements = shape.iter().product();
    let values = vec![f16::from_f32(value); num_elements];
    let layout = client.create_tensor_from_slice(f16::as_bytes(&values), shape.clone(), 2);
    TensorHandle::new(
        layout.memory,
        shape.clone(),
        Strides::new(&[1, shape[0]]),
        f16::as_type_native_unchecked(),
    )
}

fn assert_strategy_parity(
    client: &ComputeClient<CudaRuntime>,
    strategy: Strategy,
    m: usize,
    n: usize,
    k: usize,
    transposed_rhs: bool,
) {
    let lhs = filled_tensor(client, shape![m, k], 1.0);
    let rhs = match transposed_rhs {
        true => transposed_filled_tensor(client, shape![k, n], 1.0),
        false => filled_tensor(client, shape![k, n], 1.0),
    };
    let out = filled_tensor(client, shape![m, n], 0.0);
    let dtype = f16::as_type_native_unchecked().storage_type();

    for launch_index in 0..2 {
        launch_ref(
            &strategy,
            client,
            InputBinding::Normal(lhs.clone().binding(), dtype),
            InputBinding::Normal(rhs.clone().binding(), dtype),
            out.clone().binding(),
            &mut f16_dtypes(),
        )
        .unwrap_or_else(|error| panic!("{strategy} launch {launch_index} failed: {error:?}"));

        let expected = k as f32;
        let actual = read_f16(client, out.clone());
        let mismatches = actual
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != expected)
            .take(16)
            .collect::<Vec<_>>();
        assert!(
            mismatches.is_empty(),
            "{strategy} launch {launch_index} parity mismatches at {mismatches:?}; expected {expected}",
        );
    }
}

#[test]
fn test_terminalo3_naive_large_m_axis_parity() {
    let client = CudaRuntime::client(&Default::default());
    let lhs = filled_tensor(&client, shape![LARGE_M, 1], 1.0);
    let rhs = filled_tensor(&client, shape![1, 1], 1.0);
    let out = filled_tensor(&client, shape![LARGE_M, 1], 0.0);
    let dtype = f16::as_type_native_unchecked().storage_type();

    launch_ref(
        &Strategy::Naive,
        &client,
        InputBinding::Normal(lhs.binding(), dtype),
        InputBinding::Normal(rhs.binding(), dtype),
        out.clone().binding(),
        &mut f16_dtypes(),
    )
    .unwrap();

    let actual = read_f16(&client, out);
    let mismatches = actual
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 1.0)
        .take(16)
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "Naive large-axis parity mismatches at {mismatches:?}",
    );
}

#[test]
fn test_terminalo3_cyclic_cmma_mma_output_reuse_parity() {
    let client = CudaRuntime::client(&Default::default());

    for strategy in [
        Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Cmma,
        })),
        Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Mma,
        })),
        Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: true,
            tile_matmul: TileMatmulKind::Cmma,
        })),
        Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: true,
            tile_matmul: TileMatmulKind::Mma,
        })),
        Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(4),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Cmma,
        })),
        Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(4),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Mma,
        })),
        Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(8),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Cmma,
        })),
        Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(8),
            rows_per_plane: Some(2),
            tile_matmul: TileMatmulKind::Mma,
        })),
        Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: false,
            tile_matmul: TileMatmulKind::Cmma,
        })),
        Strategy::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: false,
            tile_matmul: TileMatmulKind::Mma,
        })),
        Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: true,
            tile_matmul: TileMatmulKind::Cmma,
        })),
        Strategy::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
            specialized: true,
            tile_matmul: TileMatmulKind::Mma,
        })),
        Strategy::SpecializedCyclicCmma(BlueprintStrategy::Inferred(().into())),
        Strategy::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())),
    ] {
        assert_strategy_parity(&client, strategy, 256, 512, 256, false);
    }
}

#[test]
fn test_terminalo3_cyclic_large_m_axis_parity() {
    let client = CudaRuntime::client(&Default::default());

    for strategy in [
        Strategy::SimpleCyclicCmma(Default::default()),
        Strategy::SimpleCyclicMma(Default::default()),
    ] {
        assert_strategy_parity(&client, strategy, 262_144, 32, 32, true);
    }
}

#[test]
fn test_terminalo3_tma_cmma_mma_output_reuse_parity() {
    let client = CudaRuntime::client(&Default::default());

    for strategy in [
        Strategy::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Cmma,
        })),
        Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: false,
            tile_matmul: TileMatmulKind::Mma,
        })),
        Strategy::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: true,
            tile_matmul: TileMatmulKind::Cmma,
        })),
        Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
            multi_rows: true,
            tile_matmul: TileMatmulKind::Mma,
        })),
        Strategy::SpecializedTmaCmma(BlueprintStrategy::Inferred(().into())),
        Strategy::SpecializedTmaMma(BlueprintStrategy::Inferred(().into())),
    ] {
        assert_strategy_parity(&client, strategy, 256, 512, 256, false);
    }
}
